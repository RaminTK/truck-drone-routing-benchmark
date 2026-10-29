import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from instance_generator import ProblemInstance
from decomposition import solve


def main():
    parser = argparse.ArgumentParser(description='Run the fixed decomposition heuristic or small Held-Karp reference.')
    parser.add_argument('--size', type=int, choices=(10, 20, 30, 40), default=10)
    parser.add_argument('--method', choices=('heuristic', 'exact'), default='heuristic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.method == 'exact' and args.size != 10:
        parser.error('Held-Karp reference is restricted to the evaluated small size n=10.')
    fields = ('coords', 'truck_accessible', 'demand', 'svi', 'depot', 'truck_speed', 'drone_speed', 'seed')
    results = []
    for item in json.loads((ROOT / 'data/test_instances.json').read_text()):
        if item['customer_count'] == args.size:
            result = solve(ProblemInstance(**{key: item[key] for key in fields}), args.method)
            results.append({'size': args.size, 'instance_id': item['instance_id'], **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
