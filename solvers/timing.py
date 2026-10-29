import time
from unittest.mock import patch
import numpy as np
import vns
import decomposition
from baselines import nearest_neighbor_policy
from truck_drone_env import TruckDroneEnv

BUDGETS = (.005, .020, .100, 1.0)

class Deadline(Exception):
    pass

class Clock:
    def __init__(self, budget):
        self.started = time.perf_counter()
        self.budget = budget
        self.sequence = None
        self.ready = None
        self.initialization = None

    def elapsed(self):
        return time.perf_counter() - self.started

    def check(self):
        if self.budget is not None and self.elapsed() >= self.budget:
            raise Deadline

    def publish(self, sequence):
        candidate = list(sequence)
        ready = self.elapsed()
        if self.budget is not None and ready >= self.budget:
            raise Deadline
        self.sequence, self.ready = candidate, ready
        if self.initialization is None:
            self.initialization = ready

def decomposition_search(instance, clock):
    clock.check()
    trucks, _, matrix, assignments = decomposition.geometry(instance)
    clock.check()
    remaining, route, current = set(trucks), [], -1
    while remaining:
        clock.check()
        current = min(remaining, key=lambda node: (matrix[current, node], node))
        route.append(current)
        remaining.remove(current)

    def publish():
        sequence = []
        for launch in [-1] + route:
            if launch != -1:
                sequence.append(launch)
            sequence.extend(sorted(node for node, assigned in assignments.items() if assigned == launch))
        clock.publish(sequence)

    publish()
    while True:
        best = (decomposition.length(route, matrix), tuple(route))
        for source in range(len(route)):
            reduced = route[:source] + route[source + 1:]
            for target in range(len(route)):
                clock.check()
                candidate = reduced[:target] + [route[source]] + reduced[target:]
                best = min(best, (decomposition.length(candidate, matrix), tuple(candidate)))
        for first in range(len(route)):
            for last in range(first + 1, len(route)):
                clock.check()
                candidate = route[:first] + route[first:last + 1][::-1] + route[last + 1:]
                best = min(best, (decomposition.length(candidate, matrix), tuple(candidate)))
        clock.check()
        if best[0] >= decomposition.length(route, matrix) - 1e-10:
            return
        route = list(best[1])
        publish()

def vns_search(instance, seed, clock):
    clock.check()
    config = vns.VNSConfig()
    rng = np.random.default_rng(seed)
    cache = {}
    best = list(nearest_neighbor_policy(instance).service_sequence)
    if not vns.is_feasible_sequence(best, instance.n_nodes):
        best = list(range(instance.n_nodes))
    original_objective = vns._objective_for_sequence

    def objective(**kwargs):
        clock.check()
        value = original_objective(**kwargs)
        clock.check()
        return value

    def score(sequence):
        return objective(instance=instance, sequence=sequence, max_nodes=50, objective_cache=cache)

    best_objective = score(best)
    clock.publish(best)
    iteration = no_improve = 0
    neighborhoods = vns._neighborhoods()
    with patch.object(vns, '_objective_for_sequence', objective):
        while iteration < config.max_iterations and no_improve < config.max_no_improve:
            clock.check()
            neighborhood_index, improved = 0, False
            while neighborhood_index < len(neighborhoods):
                clock.check()
                candidate = neighborhoods[neighborhood_index](best, rng)
                if not vns.is_feasible_sequence(candidate, instance.n_nodes):
                    neighborhood_index += 1
                    continue
                candidate, candidate_objective = vns._local_search(
                    instance=instance, sequence=candidate, rng=rng, max_nodes=50,
                    config=config, objective_cache=cache, start_time=clock.started)
                clock.check()
                if vns._improves(candidate_objective, best_objective):
                    clock.publish(candidate)
                    best, best_objective = candidate, candidate_objective
                    neighborhood_index, improved = 0, True
                else:
                    neighborhood_index += 1
            no_improve = 0 if improved else no_improve + 1
            iteration += 1

def measure(method, instance, model, seed, budget):
    clock = Clock(budget)
    timed_out = False
    try:
        if method == 'NN':
            clock.publish(nearest_neighbor_policy(instance).service_sequence)
        elif method == 'PPO':
            env = TruckDroneEnv(n_nodes=instance.n_nodes, max_nodes=50)
            observation, _ = env.reset(options={'instance': instance})
            for step in range(instance.n_nodes + 2):
                action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
                observation, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
            clock.publish(env.service_sequence)
        elif method == 'VNS':
            vns_search(instance, seed, clock)
        else:
            decomposition_search(instance, clock)
    except Deadline:
        timed_out = True
    elapsed = clock.elapsed()
    return {'method': method, 'budget_seconds': budget, 'runtime_seconds': elapsed,
            'overrun_seconds': max(0.0, elapsed - budget) if budget is not None else 0.0,
            'timed_out': timed_out, 'available': clock.sequence is not None,
            'incumbent_ready_seconds': clock.ready, 'initialization_seconds': clock.initialization,
            'service_sequence': clock.sequence}

def validate(instance, sequence):
    info = vns.evaluate_sequence_in_env(instance, sequence, max_nodes=50)
    assert info['feasible'] and info['invalid_actions'] == 0 and info['completion_rate'] == 1
    assert sorted(sequence) == list(range(instance.n_nodes))
    current = instance.depot.astype(np.float64)
    truck_distance = drone_distance = 0.0
    for node in sequence:
        target = instance.coords[node].astype(np.float64)
        if instance.truck_accessible[node]:
            truck_distance += float(np.abs(current - target).sum())
            current = target
        else:
            drone_distance += 2 * float(np.linalg.norm(current - target))
    expected = truck_distance / instance.truck_speed + drone_distance / instance.drone_speed
    assert np.isclose(expected, info['makespan'], atol=1e-4, rtol=1e-6)
    assert np.isclose(truck_distance + drone_distance, info['total_distance'], atol=1e-4, rtol=1e-6)
    return {key: info[key] for key in ('makespan', 'total_distance', 'completion_rate', 'invalid_actions', 'feasible')} | {'truck_distance': truck_distance, 'drone_distance': drone_distance}

class TrajectoryClock(Clock):
    def __init__(self):
        super().__init__(1.0)
        self.history = []
        self.observations = {}

    def observe(self, elapsed):
        for budget in BUDGETS:
            if elapsed >= budget and budget not in self.observations:
                self.observations[budget] = elapsed

    def check(self):
        elapsed = self.elapsed()
        self.observe(elapsed)
        if elapsed >= self.budget:
            raise Deadline

    def publish(self, sequence):
        candidate = list(sequence)
        ready = self.elapsed()
        self.observe(ready)
        if ready >= self.budget:
            raise Deadline
        self.sequence, self.ready = candidate, ready
        if self.initialization is None:
            self.initialization = ready
        self.history.append({'ready_seconds': ready, 'service_sequence': candidate})

def trajectory(method, instance, seed):
    clock = TrajectoryClock()
    timed_out = False
    try:
        if method == 'VNS':
            vns_search(instance, seed, clock)
        else:
            decomposition_search(instance, clock)
    except Deadline:
        timed_out = True
    elapsed = clock.elapsed()
    clock.observe(elapsed)
    return {'method': method, 'runtime_seconds': elapsed, 'timed_out': timed_out,
            'overrun_seconds': max(0.0, elapsed - 1.0), 'initialization_seconds': clock.initialization,
            'incumbent_history': clock.history, 'budget_observation_seconds': clock.observations}
