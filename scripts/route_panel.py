import numpy as np

def panel(axis, item, route, title):
    coords, depot = np.array(item['coords']), np.array(item['depot'])
    accessible = np.array(item['truck_accessible'], dtype=bool)
    points = {-1: depot, **dict(enumerate(coords))}
    axis.set(xlim=(-5, 105), ylim=(-5, 105), xticks=[0, 50, 100], yticks=[0, 50, 100])
    axis.set_aspect('equal')
    axis.grid(color='#e8ebee', linewidth=.55, zorder=0)
    axis.tick_params(labelsize=8, length=2, colors='#555555')
    for spine in axis.spines.values():
        spine.set_color('#c9cfd5')
    previous = depot
    for node in route['truck_route']:
        target = coords[node]
        elbow = np.array([target[0], previous[1]])
        axis.plot([previous[0], elbow[0], target[0]], [previous[1], elbow[1], target[1]], color='#1764a0', linewidth=1.5, zorder=2)
        origin = elbow if np.linalg.norm(target - elbow) > 1e-6 else previous
        axis.annotate('', xy=target, xytext=origin, arrowprops={'arrowstyle': '-|>', 'color': '#1764a0', 'lw': 1.4, 'shrinkA': 0, 'shrinkB': 4}, zorder=2)
        previous = target
    for sortie in route['sorties']:
        origin, target = points[sortie['launch']], coords[sortie['customer']]
        axis.annotate('', xy=target, xytext=origin, arrowprops={'arrowstyle': '<->', 'color': '#c36313', 'lw': 1.05, 'linestyle': (0, (3, 2)), 'shrinkA': 4, 'shrinkB': 4}, zorder=1)
    for launch in sorted({sortie['launch'] for sortie in route['sorties']}):
        axis.scatter(*points[launch], s=130, facecolors='none', edgecolors='#c36313', linewidths=1.05, zorder=4)
    axis.scatter(coords[accessible, 0], coords[accessible, 1], s=34, marker='s', c='#1764a0', edgecolors='white', linewidths=.5, zorder=5)
    axis.scatter(coords[~accessible, 0], coords[~accessible, 1], s=40, marker='^', c='#c36313', edgecolors='white', linewidths=.5, zorder=5)
    axis.scatter(*depot, marker='*', s=150, c='#20252b', edgecolors='white', linewidths=.5, zorder=6)
    axis.annotate('D', depot, xytext=(5, -12), textcoords='offset points', fontsize=8, fontweight='bold', zorder=7)
    order = {node: index + 1 for index, node in enumerate(route['truck_route'])}
    for node, point in enumerate(coords):
        label = f'{node}[{order[node]}]' if node in order else str(node)
        offset = (4, 5) if point[1] < 92 else (4, -10)
        axis.annotate(label, point, xytext=offset, textcoords='offset points', fontsize=7.2,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .80, 'pad': .3}, zorder=7)
    axis.set_title(f'{title}\nMakespan = {route["makespan"]:.2f}', fontsize=10, pad=8)
