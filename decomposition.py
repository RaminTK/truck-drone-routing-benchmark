import itertools
import math
import time
import numpy as np
from instance_generator import ProblemInstance, manhattan_distance, euclidean_distance
from truck_drone_env import TruckDroneEnv

def geometry(instance):
    trucks = [int(node) for node in np.flatnonzero(instance.truck_accessible)]
    launches = [-1] + trucks
    points = {-1: instance.depot, **dict(enumerate(instance.coords))}
    matrix = {(first, second): manhattan_distance(points[first], points[second]) for first in launches for second in trucks}
    assignments = {}
    for node in np.flatnonzero(~instance.truck_accessible):
        assignments[int(node)] = min(launches, key=lambda launch: (2 * euclidean_distance(points[launch], points[node]), launch))
    return trucks, points, matrix, assignments

def length(route, matrix):
    return sum(matrix[first, second] for first, second in zip([-1] + list(route), route))

def heuristic(trucks, matrix):
    remaining = set(trucks)
    route = []
    current = -1
    while remaining:
        current = min(remaining, key=lambda node: (matrix[current, node], node))
        route.append(current)
        remaining.remove(current)
    while True:
        best = (length(route, matrix), tuple(route))
        for source in range(len(route)):
            reduced = route[:source] + route[source + 1:]
            for target in range(len(route)):
                candidate = reduced[:target] + [route[source]] + reduced[target:]
                best = min(best, (length(candidate, matrix), tuple(candidate)))
        for first in range(len(route)):
            for last in range(first + 1, len(route)):
                candidate = route[:first] + route[first:last + 1][::-1] + route[last + 1:]
                best = min(best, (length(candidate, matrix), tuple(candidate)))
        if best[0] >= length(route, matrix) - 1e-10:
            return route
        route = list(best[1])

def exact(trucks, matrix):
    if not trucks:
        return [], 0.0
    count = len(trucks)
    costs, parents = {}, {}
    for endpoint, node in enumerate(trucks):
        costs[1 << endpoint, endpoint] = matrix[-1, node]
        parents[1 << endpoint, endpoint] = None
    for mask in range(1, 1 << count):
        for endpoint in range(count):
            if not mask & (1 << endpoint) or mask == 1 << endpoint:
                continue
            previous_mask = mask ^ (1 << endpoint)
            cost, previous = min((costs[previous_mask, candidate] + matrix[trucks[candidate], trucks[endpoint]], candidate) for candidate in range(count) if previous_mask & (1 << candidate))
            costs[mask, endpoint] = cost
            parents[mask, endpoint] = previous
    mask = (1 << count) - 1
    cost, endpoint = min((costs[mask, endpoint], endpoint) for endpoint in range(count))
    route = []
    while endpoint is not None:
        route.append(trucks[endpoint])
        previous = parents[mask, endpoint]
        mask ^= 1 << endpoint
        endpoint = previous
    return route[::-1], cost

def solve(instance, method):
    started = time.perf_counter()
    trucks, points, matrix, assignments = geometry(instance)
    solver_start = time.perf_counter()
    if method == 'exact':
        route, certified_truck = exact(trucks, matrix)
    else:
        route = heuristic(trucks, matrix)
        certified_truck = None
    solver_seconds = time.perf_counter() - solver_start
    sequence = []
    for launch in [-1] + route:
        if launch != -1:
            sequence.append(launch)
        sequence.extend(sorted(node for node, assigned in assignments.items() if assigned == launch))
    truck_distance = length(route, matrix)
    drone_distance = sum(2 * euclidean_distance(points[launch], points[node]) for node, launch in assignments.items())
    analytical_minimum = sum(min(2 * euclidean_distance(points[launch], points[node]) for launch in [-1] + trucks) for node in assignments)
    assert math.isclose(drone_distance, analytical_minimum, abs_tol=1e-8)
    env = TruckDroneEnv(n_nodes=instance.n_nodes, max_nodes=max(50, instance.n_nodes))
    env.reset(options={'instance': instance})
    current = -1
    independent_truck = independent_drone = 0.0
    for node in sequence:
        if instance.truck_accessible[node]:
            independent_truck += float(np.abs(points[current].astype(np.float64) - points[node].astype(np.float64)).sum())
            current = node
        else:
            assert assignments[node] == current
            independent_drone += 2 * float(np.linalg.norm(points[current].astype(np.float64) - points[node].astype(np.float64)))
        _, _, terminated, truncated, info = env.step(node)
        assert not truncated
    assert terminated and info['feasible'] and info['invalid_actions'] == 0
    assert sorted(sequence) == list(range(instance.n_nodes))
    assert env.truck_route == [-1] + route
    for sortie in env.drone_routes:
        assert sortie['launch_node'] == sortie['return_node'] == assignments[sortie['served_node']]
    makespan = truck_distance / instance.truck_speed + drone_distance / instance.drone_speed
    independent_makespan = independent_truck / instance.truck_speed + independent_drone / instance.drone_speed
    for observed, expected in [(env.total_distance, truck_distance + drone_distance), (env.elapsed_time, makespan), (independent_truck, truck_distance), (independent_drone, drone_distance), (independent_makespan, makespan)]:
        assert math.isclose(observed, expected, abs_tol=1e-4, rel_tol=1e-6), (observed, expected)
    if method == 'exact':
        assert certified_truck == truck_distance
    return {'method': 'decomposition_' + method, 'service_sequence': sequence, 'truck_route': route, 'launch_assignments': assignments, 'truck_customers': len(trucks), 'truck_distance': truck_distance, 'drone_distance': drone_distance, 'analytical_minimum_drone_distance': analytical_minimum, 'total_distance': truck_distance + drone_distance, 'makespan': makespan, 'environment_makespan': env.elapsed_time, 'independent_makespan': independent_makespan, 'feasible': True, 'runtime_seconds': time.perf_counter() - started, 'solver_runtime_seconds': solver_seconds, 'certified_optimal': method == 'exact', 'certified_lower_bound_makespan': makespan if method == 'exact' else None}
