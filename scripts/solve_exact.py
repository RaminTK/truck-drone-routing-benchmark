import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='Optional Gurobi solve using the fixed formulation and saved heuristic warm starts.')
    parser.add_argument('--size', type=int, choices=(10, 20, 30, 40), default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import gurobipy as gp
    from instance_generator import ProblemInstance
    from decomposition import geometry
    from solvers.gurobi import solve
    warm = {row['instance_id']: row for row in json.loads((ROOT / 'results/decomposition.json').read_text())
            if row['size'] == args.size and row['method'] == 'decomposition_heuristic'}
    fields = ('coords', 'truck_accessible', 'demand', 'svi', 'depot', 'truck_speed', 'drone_speed', 'seed')
    results = []
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for item in json.loads((ROOT / 'data/test_instances.json').read_text()):
            if item['customer_count'] != args.size:
                continue
            instance = ProblemInstance(**{key: item[key] for key in fields})
            trucks, _, matrix, _ = geometry(instance)
            initial = warm[item['instance_id']]
            task = {'size': args.size, 'instance_id': item['instance_id'], 'trucks': trucks,
                    'matrix': [[first, second, value] for (first, second), value in matrix.items()],
                    'truck_speed': instance.truck_speed, 'drone_time': initial['drone_distance'] / instance.drone_speed,
                    'warm_route': initial['truck_route'], 'warm_truck_distance': initial['truck_distance']}
            results.append(solve(task, env))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
