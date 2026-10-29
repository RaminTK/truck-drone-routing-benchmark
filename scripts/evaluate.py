import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sb3_contrib import MaskablePPO
from evaluate import evaluate_ppo_policy
from baselines import run_baselines
from instance_generator import ProblemInstance


def main():
    parser = argparse.ArgumentParser(description='Evaluate original per-size policies and valid greedy baselines; no training or search.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(1)
    models = {size: MaskablePPO.load(ROOT / f'checkpoints/original/n{size}/policy.zip', device='cpu') for size in (10, 20, 30, 40)}
    fields = ('coords', 'truck_accessible', 'demand', 'svi', 'depot', 'truck_speed', 'drone_speed', 'seed')
    results = []
    for item in json.loads((ROOT / 'data/test_instances.json').read_text()):
        size = item['customer_count']
        instance = ProblemInstance(**{key: item[key] for key in fields})
        results.append(evaluate_ppo_policy(models[size], instance, size, max_nodes=50, deterministic=True, mask_actions=True).to_record(size, item['instance_id'], item['child_seed']))
        for result in run_baselines(instance, seed=42 + 10000 * size + item['instance_id']).values():
            results.append(result.to_record(size, item['instance_id'], item['child_seed']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
