"""Heuristic baselines for the truck-drone routing study."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import json
from typing import Callable

import numpy as np

from instance_generator import (
    ProblemInstance,
    drone_round_trip_time,
    euclidean_distance,
    manhattan_distance,
    truck_travel_time,
)
from truck_drone_env import (
    TruckDroneEnv,
    DEFAULT_REWARD_CONFIG,
    DEPOT_NODE_ID,
    PENALIZED_MAKESPAN_INVALID_WEIGHT,
    PENALIZED_MAKESPAN_UNSERVED_WEIGHT,
)


@dataclass
class PolicyResult:
    method: str
    makespan: float
    total_distance: float
    completion_rate: float
    invalid_actions: int
    runtime: float
    steps: int
    served_customers: int = 0
    unserved_customers: int = 0
    total_reward: float = 0.0
    feasible: bool = False
    penalized_makespan: float = 0.0
    terminal_status: str = "unknown"
    trace: list[dict[str, float | int | str]] = field(default_factory=list)
    service_sequence: list[int] = field(default_factory=list)
    action_history: list[int] = field(default_factory=list)
    truck_route: list[int] = field(default_factory=list)
    drone_routes: list[dict[str, float | int | str]] = field(default_factory=list)
    route_signature: str = ""
    num_unique_actions: int = 0
    ppo_action_fallbacks: int = 0
    training_runtime_seconds: float = 0.0
    inference_runtime_seconds: float | None = None
    evaluation_runtime_seconds: float = 0.0

    def to_record(self, size: int, instance_id: int, seed: int | None) -> dict[str, object]:
        inference_runtime_seconds = (
            self.runtime if self.inference_runtime_seconds is None else self.inference_runtime_seconds
        )
        total_runtime_seconds = (
            self.training_runtime_seconds + inference_runtime_seconds + self.evaluation_runtime_seconds
        )
        return {
            "size": size,
            "instance_id": instance_id,
            "seed": seed,
            "method": self.method,
            "makespan": self.makespan,
            "total_distance": self.total_distance,
            "completion_rate": self.completion_rate,
            "served_customers": self.served_customers,
            "unserved_customers": self.unserved_customers,
            "feasible": self.feasible,
            "invalid_actions": self.invalid_actions,
            "runtime": self.runtime,
            "runtime_seconds": self.runtime,
            "inference_runtime_seconds": inference_runtime_seconds,
            "evaluation_runtime_seconds": self.evaluation_runtime_seconds,
            "training_runtime_seconds": self.training_runtime_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "penalized_makespan": self.penalized_makespan,
            "terminal_status": self.terminal_status,
            "route_signature": self.route_signature,
            "service_sequence": " ".join(str(node) for node in self.service_sequence),
            "truck_route": " ".join(_format_node(node) for node in self.truck_route),
            "num_unique_actions": self.num_unique_actions,
            "action_history": " ".join(str(action) for action in self.action_history),
            "drone_routes": json.dumps(self.drone_routes),
            "route_history": json.dumps(self.trace),
            "ppo_action_fallbacks": self.ppo_action_fallbacks,
        }


def rule_service_mode(
    instance: ProblemInstance,
    current_position: np.ndarray,
    node_index: int,
) -> tuple[str, float, float]:
    """Use the route-sensitive default rule from the RL environment."""

    if instance.truck_accessible[node_index]:
        distance = manhattan_distance(current_position, instance.coords[node_index])
        return "truck", distance / instance.truck_speed, distance

    distance = 2.0 * euclidean_distance(current_position, instance.coords[node_index])
    return "drone", distance / instance.drone_speed, distance


def explicit_service_mode(
    instance: ProblemInstance,
    current_position: np.ndarray,
    node_index: int,
    mode: str,
) -> tuple[str, float, float]:
    if mode == "truck":
        if not instance.truck_accessible[node_index]:
            raise ValueError("truck cannot serve a drone-only node")
        distance = manhattan_distance(current_position, instance.coords[node_index])
        return "truck", distance / instance.truck_speed, distance

    if mode == "drone":
        distance = 2.0 * euclidean_distance(current_position, instance.coords[node_index])
        return "drone", distance / instance.drone_speed, distance

    raise ValueError(f"unknown service mode: {mode}")


def _apply_service(
    instance: ProblemInstance,
    current_position: np.ndarray,
    current_node: int,
    node_index: int,
    service_mode: str,
    travel_time: float,
    distance: float,
    elapsed_time: float,
    total_distance: float,
    step_number: int,
) -> tuple[np.ndarray, int, dict[str, float | int | str]]:
    origin = current_position.copy()
    destination = instance.coords[node_index]
    next_position = destination.copy() if service_mode == "truck" else current_position.copy()
    next_node = node_index if service_mode == "truck" else current_node
    step = {
        "step": int(step_number),
        "action": int(node_index),
        "served_node": int(node_index),
        "node": int(node_index),
        "vehicle": service_mode,
        "mode": service_mode,
        "from_node": int(current_node),
        "to_node": int(node_index),
        "from_x": float(origin[0]),
        "from_y": float(origin[1]),
        "to_x": float(destination[0]),
        "to_y": float(destination[1]),
        "incremental_distance": float(distance),
        "incremental_time": float(travel_time),
        "service_time": float(travel_time),
        "distance": float(distance),
        "cumulative_distance": float(total_distance + distance),
        "current_time": float(elapsed_time + travel_time),
        "elapsed_after": float(elapsed_time + travel_time),
    }
    return next_position, next_node, step


def _finish_result(
    method: str,
    start_time: float,
    elapsed_time: float,
    total_distance: float,
    visited: np.ndarray,
    trace: list[dict[str, float | int | str]],
    service_sequence: list[int],
) -> PolicyResult:
    metric_start = time.perf_counter()
    served_customers = int(visited.sum())
    unserved_customers = int(len(visited) - served_customers)
    completion_rate = served_customers / len(visited)
    invalid_actions = 0
    feasible = completion_rate == 1.0 and invalid_actions == 0
    penalized_makespan = (
        elapsed_time
        + (PENALIZED_MAKESPAN_UNSERVED_WEIGHT * unserved_customers)
        + (PENALIZED_MAKESPAN_INVALID_WEIGHT * invalid_actions)
    )
    truck_route = [DEPOT_NODE_ID] + [int(step["node"]) for step in trace if step["mode"] == "truck"]
    drone_routes = [
        {
            "step": int(step["step"]),
            "launch_node": int(step["from_node"]),
            "served_node": int(step["node"]),
            "return_node": int(step["from_node"]),
            "distance": float(step["distance"]),
            "time": float(step["service_time"]),
        }
        for step in trace
        if step["mode"] == "drone"
    ]
    route_signature = "-".join(str(node) for node in service_sequence)
    evaluation_runtime_seconds = time.perf_counter() - metric_start
    total_runtime_seconds = time.perf_counter() - start_time
    inference_runtime_seconds = max(0.0, total_runtime_seconds - evaluation_runtime_seconds)
    return PolicyResult(
        method=method,
        makespan=float(elapsed_time),
        total_distance=float(total_distance),
        completion_rate=float(completion_rate),
        invalid_actions=invalid_actions,
        runtime=float(total_runtime_seconds),
        steps=len(service_sequence),
        served_customers=served_customers,
        unserved_customers=unserved_customers,
        total_reward=_score_trace_reward(
            trace=trace,
            completed=bool(visited.all()),
            elapsed_time=elapsed_time,
            total_distance=total_distance,
        ),
        feasible=feasible,
        penalized_makespan=float(penalized_makespan),
        terminal_status="completed" if visited.all() else "incomplete",
        trace=trace,
        service_sequence=service_sequence,
        action_history=list(service_sequence),
        truck_route=truck_route,
        drone_routes=drone_routes,
        route_signature=route_signature,
        num_unique_actions=len(set(service_sequence)),
        training_runtime_seconds=0.0,
        inference_runtime_seconds=float(inference_runtime_seconds),
        evaluation_runtime_seconds=float(evaluation_runtime_seconds),
    )


def _score_trace_reward(
    trace: list[dict[str, float | int | str]],
    completed: bool,
    elapsed_time: float,
    total_distance: float,
) -> float:
    reward = 0.0
    for step in trace:
        reward -= DEFAULT_REWARD_CONFIG.step_penalty
        reward += DEFAULT_REWARD_CONFIG.service_reward
        reward -= DEFAULT_REWARD_CONFIG.time_weight * float(step["service_time"])
        reward -= DEFAULT_REWARD_CONFIG.distance_weight * float(step["distance"])

    if completed:
        reward += DEFAULT_REWARD_CONFIG.completion_bonus
        reward -= DEFAULT_REWARD_CONFIG.final_makespan_weight * elapsed_time
        reward -= DEFAULT_REWARD_CONFIG.final_distance_weight * total_distance
    return float(reward)


def _format_node(node: int) -> str:
    return "depot" if node == DEPOT_NODE_ID else str(node)


def random_valid_policy(instance: ProblemInstance, seed: int | None = None) -> PolicyResult:
    """Choose uniformly among unvisited nodes and serve each valid choice."""

    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    current_position = instance.depot.copy()
    current_node = DEPOT_NODE_ID
    visited = np.zeros(instance.n_nodes, dtype=bool)
    elapsed_time = 0.0
    total_distance = 0.0
    trace: list[dict[str, float | int | str]] = []
    service_sequence: list[int] = []

    while not visited.all():
        candidates = np.flatnonzero(~visited)
        node_index = int(rng.choice(candidates))
        service_mode, travel_time, distance = rule_service_mode(instance, current_position, node_index)
        current_position, current_node, step = _apply_service(
            instance,
            current_position,
            current_node,
            node_index,
            service_mode,
            travel_time,
            distance,
            elapsed_time,
            total_distance,
            len(service_sequence) + 1,
        )
        elapsed_time += travel_time
        total_distance += distance
        visited[node_index] = True
        trace.append(step)
        service_sequence.append(node_index)

    return _finish_result(
        "random_valid",
        start_time,
        elapsed_time,
        total_distance,
        visited,
        trace,
        service_sequence,
    )


def nearest_neighbor_policy(instance: ProblemInstance) -> PolicyResult:
    """Greedy policy that picks the unvisited node with minimum feasible service time."""

    start_time = time.perf_counter()
    current_position = instance.depot.copy()
    current_node = DEPOT_NODE_ID
    visited = np.zeros(instance.n_nodes, dtype=bool)
    elapsed_time = 0.0
    total_distance = 0.0
    trace: list[dict[str, float | int | str]] = []
    service_sequence: list[int] = []

    while not visited.all():
        best: tuple[float, int, str, float] | None = None
        for node_index in np.flatnonzero(~visited):
            service_mode, travel_time, distance = rule_service_mode(
                instance,
                current_position,
                int(node_index),
            )
            candidate = (travel_time, int(node_index), service_mode, distance)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            break

        travel_time, node_index, service_mode, distance = best
        current_position, current_node, step = _apply_service(
            instance,
            current_position,
            current_node,
            node_index,
            service_mode,
            travel_time,
            distance,
            elapsed_time,
            total_distance,
            len(service_sequence) + 1,
        )
        elapsed_time += travel_time
        total_distance += distance
        visited[node_index] = True
        trace.append(step)
        service_sequence.append(node_index)

    return _finish_result(
        "nearest_neighbor",
        start_time,
        elapsed_time,
        total_distance,
        visited,
        trace,
        service_sequence,
    )



MAX_NODES = 50

def corrected_drone_priority_sequence(instance: ProblemInstance) -> tuple[list[int], float]:
    """Prioritize drone-only service using the shared vehicle-assignment rules."""

    start = time.perf_counter()
    env = TruckDroneEnv(n_nodes=instance.n_nodes, max_nodes=MAX_NODES)
    _, info = env.reset(options={"instance": instance})
    sequence: list[int] = []

    while not env.visited.all():
        candidates: list[tuple[int, float, int]] = []
        for node_index in np.flatnonzero(~env.visited):
            node_index = int(node_index)
            service_mode, travel_time, _ = env._choose_service_mode(node_index)
            priority = 0 if service_mode == "drone" else 1
            candidates.append((priority, float(travel_time), node_index))

        _, _, selected_node = min(candidates)
        sequence.append(selected_node)
        _, _, terminated, truncated, info = env.step(selected_node)
        if truncated:
            raise RuntimeError("Corrected Drone Priority unexpectedly truncated")
        if terminated and not env.visited.all():
            raise RuntimeError("Corrected Drone Priority terminated before serving all nodes")

    runtime_seconds = time.perf_counter() - start
    if not info.get("feasible", False):
        raise RuntimeError("Corrected Drone Priority produced an infeasible route")
    return sequence, runtime_seconds


def drone_priority_policy(instance: ProblemInstance) -> PolicyResult:
    from vns import sequence_to_policy_result
    sequence, elapsed = corrected_drone_priority_sequence(instance)
    return sequence_to_policy_result(instance=instance, sequence=sequence, max_nodes=50,
                                     runtime=elapsed, method='drone_priority_corrected')


def run_baselines(instance: ProblemInstance, seed: int | None = None,
                  include_ortools: bool = False) -> dict[str, PolicyResult]:
    return {'random_valid': random_valid_policy(instance, seed=seed),
            'nearest_neighbor': nearest_neighbor_policy(instance),
            'drone_priority_corrected': drone_priority_policy(instance)}
