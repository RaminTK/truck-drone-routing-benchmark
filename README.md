# Truck drone routing benchmark

Reproducible synthetic routing experiments with fixed vehicle assignments,
Maskable PPO, valid greedy baselines, VNS, decomposition and small-instance/exact
references. This repository has fresh Git history and contains research artifacts
only. MIT license; solver/dependency licenses remain separate.

## Get the evaluated checkpoints

The single final asset contains **64 unchanged evaluated checkpoints**: four
original curriculum policies and 60 policies from five configurations, three
independent seeds and four evaluated sizes. Every included stage checkpoint
supports a reported evaluation; unevaluated/intermediate snapshots are excluded.

Download [evaluated-checkpoints-v1.zip](https://github.com/RaminTK/truck-drone-routing-benchmark/releases/download/v1.0.0/evaluated-checkpoints-v1.zip)
and extract it at the repository root. Then:

```bash
python3 scripts/verify.py
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/plots.py --kind all --output figures
```

The verifier checks published SHA-256 values without executing policies. The
tests use one three-customer hand check, not training or a benchmark sweep.
Observed dependency versions are pinned; a fresh cross-platform installation and
bit-identical retraining are not established. The evaluated runtime used Python
3.14.3, CPU PyTorch 2.12.0, SB3/sb3-contrib 2.8.0. Exact Gurobi references used a
separate Python 3.12.2/Gurobi 12.0.0 environment; a suitable license is required.

## Layout

```
truck_drone_env.py, instance_generator.py   evaluated environment and generator
baselines.py, vns.py, decomposition.py     valid baseline implementations
train_ppo.py, evaluate.py                 original learning/evaluation routines
config/                                  frozen settings, seeds and model hashes
data/                                    fixed test/train/validation/transfer data
checkpoints/                             extracted final asset (not stored in Git)
experiments/frozen/run.py                 independent configurations and evaluation
solvers/                                 Gurobi formulation and soft-budget helpers
results/                                 clean instance-level and plotting inputs
scripts/                                 verification, evaluation and plots/tables
tests/                                   small operational-rule checks
```

## Reproduce evaluations without training

```bash
python scripts/evaluate.py --output outputs/original-evaluation.json
python experiments/frozen/run.py evaluate --configuration full --seed 42
```

The first command uses the original size-matched checkpoints and valid Random,
NN and corrected DP on the fixed 400 test instances. The second evaluates the
four saved stages for one independent configuration/seed; repeat explicitly for
other names/seeds if desired. Output paths refuse overwrites. VNS/decomposition
implementations and saved results are available without rerunning search.

Training is OPTIONAL, costly, and never needed to read the saved evidence:

```bash
python experiments/frozen/run.py train --configuration full --seed 42 --execute-full
```

Configurations: `full`, `bc_only`, `no_bc`, `no_curriculum`, `aligned_reward`;
seeds 42/43/44. Training writes only under `outputs/training`, not the distributed
checkpoints. Frozen budgets are 200k/400k/600k/800k requested PPO transitions,
rounded to 204800/401408/606208/802816 with 2048 steps x 4 environments and batch
256. Cloning uses 500 demonstrations per size, three epochs and batches of 512.
Fixed pools do not change with training seed. See config/experiment.json.

## Model and result interpretation

- Coordinates are uniform in [0,100); truck accessibility probability 0.65;
  speeds truck=1 and drone=2. Units are simulation/coordinate units, not asserted
  physical minutes or kilometers. Truck motion uses Manhattan distance.
- Vehicle assignments are fixed. A drone sortie is a Euclidean round trip from
  the current truck position, with the truck stationary; no final depot return.
- Only corrected DP is distributed. It prioritizes drone-only customers and
  executes actions through the same environment; no drone service of truck nodes.
- `results/main.csv` contains 2000 valid records: original Random/NN/PPO/VNS and
  corrected DP, each on the same 100 instances per size. Uncontrolled historical
  runtimes and the invalid former DP implementation/results are excluded.
- `results/independent.csv` contains all 6000 configuration/seed-instance rows;
  original PPO is a separate reference. Seed variability must be calculated from
  the THREE seed-level means, not 300 independent training replications.
- `results/decomposition.json` and `results/exact.json` retain solutions, bounds,
  solver status and certification. Time-limited incumbents are not called optimal.
- `results/timing` contains necessary instance-level measurements and incumbent
  trajectories, not diagnostic logs. Initialization is included; generation,
  model loading and independent validation are excluded. Budgets are SOFT;
  late incumbents are excluded. Matched means use identical available-instance
  intersections; 10/15-ms snapshots come from saved trajectories, not new runs.
- Training curves use saved episode-makespan means in nonoverlapping 8192-step
  bins, separately per seed and stage. Validation has only stage-end points over
  20 fixed instances. Problem size changes at curriculum boundaries.
- Test data were reconstructed and validated against historical route traces;
  demand/SVI were not independently archived. Geometry and vehicle assignments
  match those traces. Original hardware provenance is not fully established.
- Transfer results are exploratory from one fixed original n=40 policy at
  15/25/35/45 customers, 20 instances each—not multi-seed robustness or scalability.

The exact evaluated policy bytes and frozen data are retained. Core environment,
generator, VNS and original PPO routines are unchanged. Packaging changes are
limited to paths, valid-DP extraction and removal of unreported optional/obsolete
baseline code. Hashes and lightweight equivalence checks establish artifact
identity; no claim is made that fresh training on another machine is identical.
