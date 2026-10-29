import math
import time
import gurobipy as gp

PARAMETERS = {'Threads': 1, 'Seed': 42, 'MIPGap': 1e-06, 'TimeLimit': 10.0, 'LazyConstraints': 1, 'OutputFlag': 0}

def solve(task, env):
    started = time.perf_counter()
    trucks = task['trucks']
    vertices = [-1] + trucks
    distance = {(first, second): value for first, second, value in task['matrix']}
    speed = task['truck_speed']
    drone_time = task['drone_time']
    result = {'size': task['size'], 'instance_id': task['instance_id'], 'truck_customers': len(trucks), 'status': None, 'truck_route': task['warm_route'], 'incumbent_source': 'heuristic_fallback', 'truck_upper_bound': task['warm_truck_distance'] / speed, 'truck_lower_bound': 0.0, 'num_variables': 0, 'num_constraints': 0, 'lazy_cuts': 0, 'node_count': 0, 'solver_runtime_seconds': 0.0, 'license_error_code': None, 'callback_error': None, 'solver_has_incumbent': False}
    if len(trucks) < 2:
        value = 0.0 if not trucks else distance[-1, trucks[0]] / speed
        result.update(status='ANALYTIC_OPTIMAL', truck_upper_bound=value, truck_lower_bound=value, truck_route=trucks, incumbent_source='analytic')
    else:
        with gp.Model(env=env) as model:
            for name, value in PARAMETERS.items():
                model.setParam(name, value)
            arcs = [(first, second) for first in vertices for second in vertices if first != second]
            variables = model.addVars(arcs, vtype=gp.GRB.BINARY, name='arc')
            model.setObjective(gp.quicksum((0.0 if second == -1 else distance[first, second] / speed) * variables[first, second] for first, second in arcs), gp.GRB.MINIMIZE)
            for node in vertices:
                model.addConstr(gp.quicksum(variables[node, other] for other in vertices if other != node) == 1)
                model.addConstr(gp.quicksum(variables[other, node] for other in vertices if other != node) == 1)
            warm = [-1] + task['warm_route'] + [-1]
            selected = set(zip(warm, warm[1:]))
            for arc in arcs:
                variables[arc].Start = int(arc in selected)
            model.update()
            result.update(num_variables=model.NumVars, num_constraints=model.NumConstrs)

            def callback(active, where):
                if where != gp.GRB.Callback.MIPSOL:
                    return
                try:
                    values = active.cbGetSolution(variables)
                    successors = {first: second for (first, second), value in values.items() if value > 0.5}
                    unseen = set(vertices)
                    while unseen:
                        node = min(unseen)
                        cycle = []
                        while node in unseen:
                            unseen.remove(node)
                            cycle.append(node)
                            node = successors[node]
                        if len(cycle) < len(vertices):
                            active.cbLazy(gp.quicksum(variables[first, second] for first in cycle for second in cycle if first != second) <= len(cycle) - 1)
                            result['lazy_cuts'] += 1
                except Exception as exc:
                    result['callback_error'] = type(exc).__name__
                    active.terminate()

            try:
                model.optimize(callback)
                names = {gp.GRB.OPTIMAL: 'OPTIMAL', gp.GRB.TIME_LIMIT: 'TIME_LIMIT', gp.GRB.INTERRUPTED: 'INTERRUPTED', gp.GRB.INFEASIBLE: 'INFEASIBLE', gp.GRB.NUMERIC: 'NUMERIC', gp.GRB.SUBOPTIMAL: 'SUBOPTIMAL'}
                result.update(status=names.get(model.Status, str(model.Status)), solver_status_code=model.Status, solver_runtime_seconds=model.Runtime, node_count=model.NodeCount)
                if math.isfinite(model.ObjBound):
                    result['truck_lower_bound'] = model.ObjBound
                if model.SolCount:
                    successors = {first: second for first, second in arcs if variables[first, second].X > 0.5}
                    route = []
                    node = successors[-1]
                    while node != -1 and node not in route:
                        route.append(node)
                        node = successors[node]
                    assert node == -1 and sorted(route) == trucks
                    result.update(truck_route=route, truck_upper_bound=model.ObjVal, incumbent_source='gurobi', solver_has_incumbent=True, solver_relative_gap=model.MIPGap)
            except gp.GurobiError as exc:
                result.update(status='LICENSE_OR_SOLVER_ERROR', license_error_code=exc.errno)
    result['incumbent_makespan'] = result['truck_upper_bound'] + drone_time
    result['certified_lower_bound_makespan'] = result['truck_lower_bound'] + drone_time
    result['absolute_bound_gap'] = max(0.0, result['incumbent_makespan'] - result['certified_lower_bound_makespan'])
    result['relative_bound_gap'] = result['absolute_bound_gap'] / abs(result['incumbent_makespan']) if result['incumbent_makespan'] else 0.0
    result['certified_optimal_within_solver_tolerance'] = result['status'] in ('OPTIMAL', 'ANALYTIC_OPTIMAL') and result['callback_error'] is None
    result['drone_time_added_to_both_bounds'] = drone_time
    result['complete_worker_seconds'] = time.perf_counter() - started
    return result
