import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from baselines import corrected_drone_priority_sequence
from instance_generator import ProblemInstance
from truck_drone_env import TruckDroneEnv


class TestCore(unittest.TestCase):
    def test_corrected_DP_shared_rules(self):
        instance = ProblemInstance(coords=np.array([[3, 0], [0, 4], [4, 0]]),
                                   truck_accessible=np.array([True, False, True]),
                                   demand=np.ones(3), svi=np.zeros(3), depot=np.zeros(2),
                                   truck_speed=1.0, drone_speed=2.0, seed=7)
        sequence, _ = corrected_drone_priority_sequence(instance)
        self.assertEqual(sequence, [1, 0, 2])
        env = TruckDroneEnv(n_nodes=3, max_nodes=50)
        env.reset(options={'instance': instance})
        for node in sequence:
            _, _, terminated, truncated, info = env.step(node)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['makespan'], 8.0)
        self.assertEqual(info['total_distance'], 12.0)
        self.assertEqual(info['invalid_actions'], 0)

    def test_fixed_data_and_checkpoint_coverage(self):
        instances = json.loads((ROOT / 'data/test_instances.json').read_text())
        self.assertEqual(len(instances), 400)
        self.assertEqual(len({(row['customer_count'], row['instance_id']) for row in instances}), 400)
        checkpoints = json.loads((ROOT / 'config/checkpoints.json').read_text())
        self.assertEqual(len(checkpoints), 64)
        for row in checkpoints:
            self.assertTrue((ROOT / row['path']).is_file())


if __name__ == '__main__':
    unittest.main()
