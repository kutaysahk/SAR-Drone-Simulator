"""
monitoring/evaluator.py
=======================
Pillar 4 – Monitoring & Evaluation
------------------------------------
Provides continuous monitoring of the drone agent during a simulation run
and rigorous post-run mathematical evaluation of its performance.

Metrics captured
----------------
* Exploration efficiency   – fraction of traversable cells visited per step
* Survivor recall          – fraction of survivors found (precision/recall/F1)
* Path optimality ratio    – actual steps taken vs BFS lower-bound
* Battery efficiency       – survivors found per 100 battery units consumed
* Knowledge-base growth    – new facts inferred per step
* Decision utility score   – mean E[U] of moves actually taken
* Hazard proximity index   – mean distance to nearest hazard when moving
* Coverage rate            – cells newly revealed per step (rolling window)

All metrics are time-stamped (step index) so they can be plotted or logged.
"""

from __future__ import annotations

import math
import time
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One row in the per-step telemetry log."""
    step: int
    timestamp: float
    drone_x: int
    drone_y: int
    battery_remaining: int
    battery_consumed: int
    cells_revealed: int
    total_explored: int
    traversable_total: int
    exploration_fraction: float
    survivors_found: int
    kb_fact_count: int
    kb_new_facts: int
    decision_utility: float
    hazard_proximity: float
    coverage_rate: float          # rolling 10-step average cells/step


@dataclass
class EvaluationReport:
    """Final mathematical evaluation produced at end of run."""
    # --- mission outcome ---
    total_survivors: int
    survivors_found: int
    survivor_recall: float        # TP / (TP + FN)
    survivor_precision: float     # TP / (TP + FP)  – beacons vs real
    survivor_f1: float

    # --- efficiency ---
    total_steps: int
    total_cells_explored: int
    traversable_cells: int
    exploration_efficiency: float  # explored / traversable
    path_optimality_ratio: float   # BFS lower-bound steps / actual steps (≤1 ideal)
    battery_consumed: int
    battery_efficiency: float      # survivors_found / (battery_consumed / 100)

    # --- knowledge & reasoning ---
    final_kb_size: int
    total_inferences: int
    inference_rate: float          # inferences / step

    # --- decision quality ---
    mean_decision_utility: float
    std_decision_utility: float
    mean_hazard_proximity: float

    # --- coverage dynamics ---
    mean_coverage_rate: float      # cells revealed / step
    peak_coverage_rate: float
    coverage_acceleration: float   # linear regression slope of cumulative coverage

    # --- optimisation feedback ---
    recommended_weight_adjustments: Dict[str, float]
    optimization_score: float      # composite [0, 1]

    # --- timing ---
    wall_time_seconds: float


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class MissionMonitor:
    """
    Attach to a simulation run and call `record_step` once per drone move.
    Call `evaluate` at the end to get a full EvaluationReport.
    """

    ROLLING_WINDOW = 10   # steps for rolling coverage rate

    def __init__(self, environment, drone, optimized_weights: list):
        self._env = environment
        self._drone = drone
        self._weights = list(optimized_weights)

        # Count traversable cells once – static for the run
        self._traversable_total = int(np.sum(
            (environment.grid == 0) | (environment.grid == 3)
        ))

        # Per-step telemetry
        self.step_log: List[StepRecord] = []

        # Internal accumulators
        self._step = 0
        self._start_time = time.monotonic()
        self._prev_explored = 0
        self._prev_kb_size = 0
        self._prev_battery = drone.battery
        self._recent_reveals: deque[int] = deque(maxlen=self.ROLLING_WINDOW)
        self._utility_history: List[float] = []

        # Hazard map (cells == 2)
        hx, hy = np.where(environment.grid == 2)
        self._hazard_positions: List[Tuple[int, int]] = list(zip(hx.tolist(), hy.tolist()))

        # Track which survivors were beaconed (for precision)
        self._beaconed_positions: set = set()

    # ------------------------------------------------------------------
    # Called every step from the simulation loop
    # ------------------------------------------------------------------

    def record_step(self,
                    next_x: int,
                    next_y: int,
                    decision_utility: Optional[float] = None):
        """
        Record telemetry for one move.

        Parameters
        ----------
        next_x, next_y  : target cell the drone is about to move to
        decision_utility: E[U] value used for this move (optional; computed
                          internally if not supplied)
        """
        self._step += 1
        drone = self._drone
        env = self._env

        # Battery consumed this step
        battery_consumed_total = self._drone.battery  # snapshot before move
        # We record the snapshot *before* move – difference will be computed
        # using previous snapshot
        battery_delta = self._prev_battery - drone.battery
        self._prev_battery = drone.battery

        # Exploration
        explored = int(np.sum(drone.internal_map != -1))
        new_cells = explored - self._prev_explored
        self._prev_explored = explored
        exploration_fraction = explored / self._traversable_total if self._traversable_total else 0.0

        # KB growth
        kb_size = len(drone.kb.facts)
        new_facts = kb_size - self._prev_kb_size
        self._prev_kb_size = kb_size

        # Decision utility
        if decision_utility is None:
            decision_utility = drone.calculate_expected_utility(
                next_x, next_y, env, self._weights
            )
        self._utility_history.append(decision_utility)

        # Hazard proximity – Euclidean distance to nearest hazard
        hazard_prox = self._nearest_hazard_distance(drone.x, drone.y)

        # Rolling coverage rate
        self._recent_reveals.append(new_cells)
        coverage_rate = float(np.mean(self._recent_reveals))

        # Survivors beaconed so far
        self._beaconed_positions.update(drone.rescue_beacons.keys())

        record = StepRecord(
            step=self._step,
            timestamp=time.monotonic() - self._start_time,
            drone_x=drone.x,
            drone_y=drone.y,
            battery_remaining=drone.battery,
            battery_consumed=battery_delta,
            cells_revealed=new_cells,
            total_explored=explored,
            traversable_total=self._traversable_total,
            exploration_fraction=exploration_fraction,
            survivors_found=len(drone.rescue_beacons),
            kb_fact_count=kb_size,
            kb_new_facts=new_facts,
            decision_utility=decision_utility,
            hazard_proximity=hazard_prox,
            coverage_rate=coverage_rate,
        )
        self.step_log.append(record)
        self._print_live(record)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationReport:
        """
        Compute and return a full EvaluationReport.
        Should be called once after the simulation loop ends.
        """
        drone = self._drone
        env = self._env
        elapsed = time.monotonic() - self._start_time

        total_survivors = len(env.survivor_locations)
        beacons = set(drone.rescue_beacons.keys())

        # --- Survivor metrics ---
        true_positives = len(beacons & env.survivor_locations)
        false_positives = len(beacons - env.survivor_locations)
        false_negatives = len(env.survivor_locations - beacons)

        recall = true_positives / total_survivors if total_survivors else 0.0
        precision = true_positives / len(beacons) if beacons else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # --- Exploration ---
        explored = int(np.sum(drone.internal_map != -1))
        exploration_efficiency = explored / self._traversable_total if self._traversable_total else 0.0

        # --- Battery ---
        battery_consumed = self._drone.battery  # remaining
        initial_battery = drone.battery + self._step  # approximate (1/step)
        # Re-derive: first record gives us a snapshot
        battery_consumed_total = (
            self.step_log[0].battery_remaining + self.step_log[0].battery_consumed
            if self.step_log else 0
        )
        # Use first log entry's prior battery
        battery_consumed_total = (
            (self.step_log[0].battery_remaining + self.step_log[0].battery_consumed)
            - self.step_log[-1].battery_remaining
        ) if self.step_log else 0

        battery_efficiency = (
            true_positives / (battery_consumed_total / 100)
            if battery_consumed_total > 0 else 0.0
        )

        # --- Path optimality ---
        # BFS lower-bound: straight-line steps if grid were clear
        path_optimality = self._compute_path_optimality()

        # --- KB / inference ---
        final_kb = len(drone.kb.facts)
        total_inferences = len(drone.kb.inference_trace)
        inference_rate = total_inferences / self._step if self._step else 0.0

        # --- Decision utility stats ---
        u = np.array(self._utility_history) if self._utility_history else np.array([0.0])
        mean_u = float(np.mean(u))
        std_u = float(np.std(u))

        # --- Hazard proximity ---
        haz_vals = [r.hazard_proximity for r in self.step_log]
        mean_haz = float(np.mean(haz_vals)) if haz_vals else 0.0

        # --- Coverage dynamics ---
        cov_rates = [r.coverage_rate for r in self.step_log]
        mean_cov = float(np.mean(cov_rates)) if cov_rates else 0.0
        peak_cov = float(np.max(cov_rates)) if cov_rates else 0.0
        coverage_accel = self._coverage_slope()

        # --- Optimization feedback ---
        weight_adjustments, opt_score = self._optimization_feedback(
            recall, exploration_efficiency, mean_u, mean_haz, path_optimality
        )

        report = EvaluationReport(
            total_survivors=total_survivors,
            survivors_found=true_positives,
            survivor_recall=recall,
            survivor_precision=precision,
            survivor_f1=f1,
            total_steps=self._step,
            total_cells_explored=explored,
            traversable_cells=self._traversable_total,
            exploration_efficiency=exploration_efficiency,
            path_optimality_ratio=path_optimality,
            battery_consumed=battery_consumed_total,
            battery_efficiency=battery_efficiency,
            final_kb_size=final_kb,
            total_inferences=total_inferences,
            inference_rate=inference_rate,
            mean_decision_utility=mean_u,
            std_decision_utility=std_u,
            mean_hazard_proximity=mean_haz,
            mean_coverage_rate=mean_cov,
            peak_coverage_rate=peak_cov,
            coverage_acceleration=coverage_accel,
            recommended_weight_adjustments=weight_adjustments,
            optimization_score=opt_score,
            wall_time_seconds=elapsed,
        )

        self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_hazard_distance(self, x: int, y: int) -> float:
        if not self._hazard_positions:
            return float(self._env.width + self._env.height)
        dists = [math.hypot(x - hx, y - hy) for hx, hy in self._hazard_positions]
        return min(dists)

    def _compute_path_optimality(self) -> float:
        """
        Approximates optimality as:
          (Manhattan distance from start to end) / actual steps taken
        A ratio of 1.0 means the drone moved in a perfectly straight line;
        lower values indicate detour overhead.
        """
        if not self.step_log or self._step == 0:
            return 1.0
        first = self.step_log[0]
        last = self.step_log[-1]
        manhattan = abs(last.drone_x - 0) + abs(last.drone_y - 0)  # start is (0,0)
        return min(1.0, manhattan / self._step) if self._step else 1.0

    def _coverage_slope(self) -> float:
        """
        Linear regression slope of cumulative explored cells vs step index.
        Positive slope = expanding coverage; near-zero = stalled.
        """
        if len(self.step_log) < 2:
            return 0.0
        steps = np.array([r.step for r in self.step_log], dtype=float)
        cumulative = np.array([r.total_explored for r in self.step_log], dtype=float)
        # Detrend: fit y = a*x + b, return a
        A = np.vstack([steps, np.ones(len(steps))]).T
        result = np.linalg.lstsq(A, cumulative, rcond=None)
        return float(result[0][0])

    def _optimization_feedback(
        self,
        recall: float,
        exploration_eff: float,
        mean_utility: float,
        mean_hazard_prox: float,
        path_optimality: float,
    ) -> Tuple[Dict[str, float], float]:
        """
        Analyses performance metrics and suggests adjustments to the GA weight
        vector [w_explore, w_probability, w_distance].

        Returns
        -------
        adjustments : dict  – suggested delta per weight dimension
        score       : float – composite performance score in [0, 1]
        """
        adjustments: Dict[str, float] = {}

        # 1. If recall is low, boost survivor-probability weight
        if recall < 0.5:
            adjustments["w_probability"] = +0.15
        elif recall > 0.85:
            adjustments["w_probability"] = -0.05  # already good; free up budget

        # 2. If exploration efficiency is low, boost exploration weight
        if exploration_eff < 0.6:
            adjustments["w_explore"] = +0.10
        elif exploration_eff > 0.9:
            adjustments["w_explore"] = -0.05

        # 3. If path optimality is poor (lots of detours), boost distance weight
        if path_optimality < 0.3:
            adjustments["w_distance"] = +0.10
        elif path_optimality > 0.7:
            adjustments["w_distance"] = -0.05

        # 4. Hazard proximity: if drone stays too close to fire, no weight fix
        #    but we flag it as a safety concern
        safety_bonus = min(1.0, mean_hazard_prox / 5.0)

        # Composite optimisation score (weighted harmonic-style combination)
        components = {
            "recall":           (recall,           0.30),
            "exploration":      (exploration_eff,  0.25),
            "path_optimality":  (path_optimality,  0.15),
            "safety":           (safety_bonus,     0.15),
            "utility_quality":  (min(1.0, max(0.0, mean_utility + 0.5)), 0.15),
        }
        score = sum(v * w for v, w in components.values())

        return adjustments, round(score, 4)

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    LIVE_INTERVAL = 25  # print live summary every N steps

    def _print_live(self, r: StepRecord):
        if r.step % self.LIVE_INTERVAL != 0:
            return
        bar_len = 20
        filled = int(r.exploration_fraction * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"[Monitor] Step {r.step:4d} | "
            f"Explored [{bar}] {r.exploration_fraction*100:5.1f}% | "
            f"Survivors {r.survivors_found} | "
            f"Battery {r.battery_remaining:4d} | "
            f"KB facts {r.kb_fact_count:4d} | "
            f"Cov.rate {r.coverage_rate:4.1f} cells/step"
        )

    def _print_report(self, rep: EvaluationReport):
        divider = "=" * 62
        print(f"\n{divider}")
        print("  PROJECT AEGIS — MISSION EVALUATION REPORT")
        print(divider)

        print("\n  ── SURVIVOR RESCUE ──")
        print(f"     Total survivors        : {rep.total_survivors}")
        print(f"     Found (true positives) : {rep.survivors_found}")
        print(f"     Recall                 : {rep.survivor_recall*100:.1f}%")
        print(f"     Precision              : {rep.survivor_precision*100:.1f}%")
        print(f"     F1 Score               : {rep.survivor_f1:.4f}")

        print("\n  ── EXPLORATION EFFICIENCY ──")
        print(f"     Total steps            : {rep.total_steps}")
        print(f"     Cells explored         : {rep.total_cells_explored} / {rep.traversable_cells}")
        print(f"     Exploration efficiency : {rep.exploration_efficiency*100:.1f}%")
        print(f"     Path optimality ratio  : {rep.path_optimality_ratio:.4f}")
        print(f"     Battery consumed       : {rep.battery_consumed}")
        print(f"     Battery efficiency     : {rep.battery_efficiency:.4f} survivors/100 units")

        print("\n  ── KNOWLEDGE & REASONING ──")
        print(f"     Final KB size          : {rep.final_kb_size} facts")
        print(f"     Total inferences       : {rep.total_inferences}")
        print(f"     Inference rate         : {rep.inference_rate:.2f} inferences/step")

        print("\n  ── DECISION QUALITY ──")
        print(f"     Mean E[U]              : {rep.mean_decision_utility:.4f}")
        print(f"     Std  E[U]              : {rep.std_decision_utility:.4f}")
        print(f"     Mean hazard proximity  : {rep.mean_hazard_proximity:.2f} cells")

        print("\n  ── COVERAGE DYNAMICS ──")
        print(f"     Mean coverage rate     : {rep.mean_coverage_rate:.2f} cells/step")
        print(f"     Peak coverage rate     : {rep.peak_coverage_rate:.2f} cells/step")
        print(f"     Coverage acceleration  : {rep.coverage_acceleration:.4f} cells/step²")

        print("\n  ── OPTIMISATION FEEDBACK ──")
        print(f"     Composite score        : {rep.optimization_score:.4f}  (0=worst, 1=best)")
        if rep.recommended_weight_adjustments:
            print("     Recommended GA weight adjustments:")
            for k, v in rep.recommended_weight_adjustments.items():
                arrow = "▲" if v > 0 else "▼"
                print(f"       {arrow} {k:20s}  Δ = {v:+.2f}")
        else:
            print("     Weights are well-tuned. No adjustments recommended.")

        print(f"\n  Wall-clock time          : {rep.wall_time_seconds:.1f}s")
        print(divider + "\n")


# ---------------------------------------------------------------------------
# JSON export helper (optional)
# ---------------------------------------------------------------------------

def export_report_json(report: EvaluationReport, path: str = "mission_report.json"):
    """Serialise the EvaluationReport to a JSON file for external analysis."""
    import dataclasses
    data = dataclasses.asdict(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Monitor] Report exported → {path}")
