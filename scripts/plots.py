import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COLORS = {'nearest_neighbor': '#1764a0', 'ppo': '#c36313', 'vns': '#16806a', 'drone_priority_corrected': '#86519c'}
LABELS = {'nearest_neighbor': 'NN', 'ppo': 'Original PPO', 'vns': 'Full-search VNS', 'drone_priority_corrected': 'Corrected DP'}


def quality(frame, output):
    figure, axis = plt.subplots(figsize=(7, 4.7))
    for method, color in COLORS.items():
        group = frame[frame.method == method].groupby('size').agg(distance=('total_distance', 'mean'), makespan=('makespan', 'mean'), sd=('makespan', 'std'))
        axis.errorbar(group.distance, group.makespan, yerr=group.sd, marker='o', color=color, capsize=3, label=LABELS[method])
        for size, row in group.iterrows():
            axis.annotate(str(size), (row.distance, row.makespan), xytext=(4, 4), textcoords='offset points', fontsize=7)
    axis.set(xlabel='Mean total distance (coordinate units)', ylabel='Mean makespan (simulation time units)')
    axis.legend(frameon=False)
    axis.grid(alpha=.2)
    figure.tight_layout()
    export(figure, output / 'quality_distance')


def learning(output):
    training = pd.read_csv(ROOT / 'results/learning/training.csv')
    validation = pd.read_csv(ROOT / 'results/learning/validation.csv')
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    bounds = [0, 204800, 606208, 1212416, 2015232]
    for seed, color, marker in zip((42, 43, 44), COLORS.values(), ('o', 's', '^')):
        for size in (10, 20, 30, 40):
            group = training[(training.seed == seed) & (training.customer_count == size)]
            axes[0].plot(group.PPO_transitions, group.mean_recorded_training_makespan, color=color, lw=1, label=f'Seed {seed}' if size == 10 else None)
        group = validation[validation.seed == seed]
        axes[1].scatter(group.PPO_transitions, group.mean_recorded_validation_makespan, color=color, marker=marker, s=30, label=f'Seed {seed}')
    for axis in axes:
        for boundary in bounds[1:-1]:
            axis.axvline(boundary, color='gray', ls='--', lw=.7)
        axis.legend(frameon=False, ncol=3)
        axis.grid(alpha=.2)
    for index, size in enumerate((10, 20, 30, 40)):
        axes[0].text((bounds[index] + bounds[index + 1]) / 2, 1.02, f'n={size}', transform=axes[0].get_xaxis_transform(), ha='center')
    axes[0].set_ylabel('Training makespan\n8192-transition bin mean')
    axes[1].set_ylabel('Stage-end validation makespan')
    axes[1].set_xlabel('Cumulative PPO transitions (cloning excluded)')
    figure.text(.5, .01, 'Customer count changes at stage boundaries. Validation: 20 instances per stage; no intermediate observations.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .98))
    export(figure, output / 'learning')


def routes(frame, output):
    from route_panel import panel
    pairs = frame[(frame['size'] == 20) & frame.method.isin(('ppo', 'nearest_neighbor'))].pivot(index='instance_id', columns='method', values='makespan')
    pairs['difference'] = pairs.ppo - pairs.nearest_neighbor
    ids = []
    for better in (True, False):
        group = pairs[pairs.difference < -1e-6] if better else pairs[pairs.difference > 1e-6]
        identifier = min(group.index, key=lambda value: (abs(group.loc[value, 'difference'] - group.difference.median()), value))
        ids.append(int(identifier))
    instances = {row['instance_id']: row for row in json.loads((ROOT / 'data/test_instances.json').read_text()) if row['customer_count'] == 20}
    decomposition = {(row['size'], row['instance_id']): row for row in json.loads((ROOT / 'results/decomposition.json').read_text()) if row['method'] == 'decomposition_heuristic'}
    figure, axes = plt.subplots(2, 4, figsize=(16, 9))
    for index, identifier in enumerate(ids):
        for column, method in enumerate(('nearest_neighbor', 'ppo', 'vns', 'decomposition')):
            if method == 'decomposition':
                row = decomposition[20, identifier]
                sequence, makespan = row['service_sequence'], row['makespan']
            else:
                row = frame[(frame['size'] == 20) & (frame.instance_id == identifier) & (frame.method == method)].iloc[0]
                sequence, makespan = json.loads(row.service_sequence), row.makespan
            item = instances[identifier]
            truck, sorties, launch = [], [], -1
            for node in sequence:
                if item['truck_accessible'][node]:
                    truck.append(node)
                    launch = node
                else:
                    sorties.append({'launch': launch, 'customer': node})
            panel(axes[index, column], item, {'truck_route': truck, 'sorties': sorties, 'makespan': makespan}, LABELS.get(method, 'Decomposition'))
        axes[index, 0].set_ylabel(f'Instance {identifier}')
    figure.text(.5, .02, 'Blue squares/truck; orange triangles/drone; star/depot; orange rings/launch=recovery. Labels: ID[truck order].\nDashed double arrows: drone round trips. Manhattan elbows are illustrative; only endpoint L1 distances are modeled.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, 1))
    export(figure, output / 'routes')
    (output / 'selected_route_ids.json').write_text(json.dumps(ids))


def export(figure, path):
    figure.savefig(path.with_suffix('.pdf'))
    figure.savefig(path.with_suffix('.png'), dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Generate tables/figures from saved results only.')
    parser.add_argument('--kind', choices=('tables', 'quality', 'routes', 'learning', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams['pdf.fonttype'] = 42
    frame = pd.read_csv(ROOT / 'results/main.csv')
    if args.kind in ('tables', 'all'):
        frame.groupby(['size', 'method']).agg(count=('makespan', 'size'), mean_makespan=('makespan', 'mean'), sd_makespan=('makespan', 'std'), truck_distance=('truck_distance', 'mean'), drone_distance=('drone_distance', 'mean')).to_csv(args.output / 'main_summary.csv')
        independent = pd.read_csv(ROOT / 'results/independent.csv')
        seed_means = independent.groupby(['configuration', 'training_seed', 'size']).makespan.mean().rename('seed_mean')
        seed_means.to_csv(args.output / 'seed_means.csv')
        seed_means.groupby(['configuration', 'size']).agg(['mean', 'std']).to_csv(args.output / 'independent_summary.csv')
    if args.kind in ('quality', 'all'):
        quality(frame, args.output)
    if args.kind in ('routes', 'all'):
        routes(frame, args.output)
    if args.kind in ('learning', 'all'):
        learning(args.output)


if __name__ == '__main__':
    main()
