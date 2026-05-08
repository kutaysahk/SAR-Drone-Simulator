"""
state/drone.py
==============
DroneState – perception, reasoning, and movement for Project Aegis.

Improvements over original
---------------------------
LOGIC (Pillar 1)
  * Hazard facts are now added to the KB when the camera sees fire (grid==2)
    or rubble (grid==1), enabling the contradiction guard in KnowledgeBase to
    prevent those cells from ever being marked Safe.
  * All 8 neighbours in the 3×3 sensor footprint that are observed as clear
    get Clear_x_y facts and Safe rules, not just the 4 cardinal ones.
  * Battery drain is now applied on every move (1 unit per step), and the
    drone deactivates when it runs out.

MATH OF AI (Pillar 2)
  * Distance feature fixed: was measured from grid centre; now measured from
    the drone's current position — the only distance that matters for routing.
  * Feature vector extended to 5 dimensions:
      x = [is_unexplored, P(survivor), drone_distance,
           neighbour_unknown_fraction, hazard_proximity_penalty]
    — neighbour_unknown_fraction rewards frontiers that open up large unseen
      areas; hazard_proximity_penalty discourages moves near fire.
  * Bayesian survivor probability update: each time a cell is observed and
    contains no survivor, its probability is multiplied by 0.1 (likelihood
    ratio for "no signal given survivor present" vs "no signal given empty").
    Nearby unseen cells receive a small probability boost when a survivor IS
    found (search-area correlation).
  * Weight vector is now 5-dimensional, matching the feature vector.
    The GA evolves all 5 weights; sign handling is moved into the feature
    definition (hazard and distance are already negated before the dot product).

PATHFINDING
  * BFS path objects are replaced with (node, parent) back-pointer dicts,
    removing the O(path_length) list-copy per enqueued node and making the
    BFS O(V+E) in both time and space.
"""

from __future__ import annotations

from collections import deque
import numpy as np
from logic.knowledge_base import KnowledgeBase


# Number of feature dimensions — must match GeneticOptimizer chromosome length
FEATURE_DIM = 5


class DroneState:
    def __init__(self, start_x: int, start_y: int, max_battery: int,
                 env_width: int, env_height: int):
        self.x = start_x
        self.y = start_y
        self.battery = max_battery
        self.is_active = True

        self.env_width = env_width
        self.env_height = env_height
        self.internal_map = np.full((env_width, env_height), -1, dtype=int)

        # PILLAR 1: Logic KB
        self.kb = KnowledgeBase()
        self.rescue_beacons: dict[tuple[int,int], float] = {}

        # Drone's own survivor probability layer (Bayesian, Pillar 2)
        # Initialised to a small uniform prior; updated from observations.
        self._survivor_belief = np.full((env_width, env_height), 0.05)

        self.mark_safe(self.x, self.y)

    # ------------------------------------------------------------------
    # Logic helpers
    # ------------------------------------------------------------------

    def mark_safe(self, x: int, y: int):
        """Update both the internal map and the Logic KB."""
        if 0 <= x < self.env_width and 0 <= y < self.env_height:
            self.internal_map[x, y] = max(self.internal_map[x, y], 0)
            self.kb.add_fact(f"Safe_{x}_{y}")

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def sense_environment(self, actual_env_state) -> list[tuple[int,int]]:
        """
        Camera sensor: reveal the immediate 3×3 area.
        Update the KB with Clear/Blocked/Heat facts for every observed cell.
        Apply Bayesian belief update to survivor probabilities.
        """
        changed_cells: list[tuple[int,int]] = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                if not (0 <= nx < self.env_width and 0 <= ny < self.env_height):
                    continue

                actual_val = actual_env_state.get_cell_truth(nx, ny)
                if self.internal_map[nx, ny] != actual_val:
                    changed_cells.append((nx, ny))
                self.internal_map[nx, ny] = actual_val

                # --- KB facts from camera observation ---
                if actual_val == 0:   # clear
                    self.kb.add_fact(f"Clear_{nx}_{ny}")
                    self.kb.add_fact(f"Safe_{nx}_{ny}")
                    # Propagate safety to ALL 8 neighbours of a clear cell
                    for ddx in [-1, 0, 1]:
                        for ddy in [-1, 0, 1]:
                            nnx, nny = nx + ddx, ny + ddy
                            if (0 <= nnx < self.env_width and
                                    0 <= nny < self.env_height):
                                self.kb.add_rule(
                                    [f"Clear_{nx}_{ny}"], f"Safe_{nnx}_{nny}"
                                )
                elif actual_val == 1:  # rubble / obstacle
                    self.kb.add_fact(f"Blocked_{nx}_{ny}")
                elif actual_val == 2:  # fire / hazard
                    self.kb.add_fact(f"Heat_{nx}_{ny}")
                    self.kb.add_fact(f"Blocked_{nx}_{ny}")
                elif actual_val == 3:  # survivor cell (traversable)
                    self.kb.add_fact(f"Clear_{nx}_{ny}")
                    self.kb.add_fact(f"Safe_{nx}_{ny}")

                # --- Survivor detection ---
                if actual_env_state.has_survivor(nx, ny):
                    if (nx, ny) not in self.rescue_beacons:
                        confidence = actual_env_state.detection_confidence(nx, ny)
                        self.rescue_beacons[(nx, ny)] = confidence
                        tag = ("Possible survivor" if confidence < 1.0
                               else "Survivor found")
                        print(f"{tag} at ({nx}, {ny})! Calling for help.")
                    # Bayesian boost: nearby unseen cells more likely to have
                    # survivors (disaster-zone clustering correlation)
                    self._bayesian_survivor_boost(nx, ny)
                else:
                    # No survivor observed → downgrade belief for this cell
                    self._survivor_belief[nx, ny] *= 0.1
                    self._survivor_belief[nx, ny] = max(
                        1e-4, self._survivor_belief[nx, ny]
                    )

        self.kb.infer()
        return changed_cells

    def _bayesian_survivor_boost(self, found_x: int, found_y: int):
        """
        When a survivor is found, modestly boost belief in unseen neighbours
        (survivors tend to cluster near collapse epicentres).
        """
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = found_x + dx, found_y + dy
                if not (0 <= nx < self.env_width and 0 <= ny < self.env_height):
                    continue
                if self.internal_map[nx, ny] == -1:  # unseen only
                    dist = max(1, abs(dx) + abs(dy))
                    self._survivor_belief[nx, ny] = min(
                        0.95,
                        self._survivor_belief[nx, ny] + 0.15 / dist
                    )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_to(self, target_x: int, target_y: int) -> bool:
        """Move only when propositional logic proves the target is Safe."""
        if self.kb.ask(f"Safe_{target_x}_{target_y}"):
            self.x = target_x
            self.y = target_y
            self.battery -= 1
            if self.battery <= 0:
                self.is_active = False
                print("Battery depleted. Drone deactivated.")
            return True
        else:
            print(f"Logic gate: refuse move to ({target_x}, {target_y}) — not proven Safe.")
            return False

    def move(self, dx: int, dy: int) -> bool:
        return self.move_to(self.x + dx, self.y + dy)

    # ------------------------------------------------------------------
    # PILLAR 2 — Feature vector and Expected Utility
    # ------------------------------------------------------------------

    def get_decision_features(self, target_x: int, target_y: int,
                               env_state) -> np.ndarray:
        """
        Returns a 5-dimensional feature vector:

          x[0]  is_unexplored            – 1 if cell unseen, else 0
          x[1]  P(survivor)              – Bayesian belief for this cell
          x[2]  drone_distance           – normalised L2 from *drone* (not centre)
          x[3]  neighbour_unknown_frac   – fraction of 3×3 footprint unseen
          x[4]  hazard_proximity_penalty – 1/(1+dist_to_nearest_fire)

        All features are in [0, 1].
        """
        # Distance from DRONE (corrected from original grid-centre bug)
        drone_vec = np.array([self.x, self.y], dtype=float)
        target_vec = np.array([target_x, target_y], dtype=float)
        max_dist = np.linalg.norm(
            np.array([self.env_width, self.env_height], dtype=float)
        )
        drone_distance = float(
            np.linalg.norm(target_vec - drone_vec) / max_dist
        ) if max_dist else 0.0

        # Exploration status
        if self.internal_map[target_x, target_y] == -1:
            is_unexplored = 1.0
            survivor_prob = float(self._survivor_belief[target_x, target_y])
        else:
            is_unexplored = 0.0
            survivor_prob = 0.0

        # Fraction of 3×3 sensor footprint still unseen
        sensor = self._sensor_cells(target_x, target_y)
        unknown_count = sum(
            1 for sx, sy in sensor if self.internal_map[sx, sy] == -1
        )
        neighbour_unknown_frac = unknown_count / len(sensor) if sensor else 0.0

        # Hazard proximity penalty
        hazard_penalty = self._hazard_proximity_penalty(target_x, target_y)

        return np.array([
            is_unexplored,
            survivor_prob,
            drone_distance,
            neighbour_unknown_frac,
            hazard_penalty,
        ], dtype=float)

    def _hazard_proximity_penalty(self, x: int, y: int) -> float:
        """
        Returns a value in (0, 1] based on proximity to known fire cells.
        Closer to fire → higher penalty.  Zero hazards → 0.
        """
        hazard_cells = [
            (hx, hy)
            for hx in range(max(0, x - 4), min(self.env_width, x + 5))
            for hy in range(max(0, y - 4), min(self.env_height, y + 5))
            if self.internal_map[hx, hy] == 2
        ]
        if not hazard_cells:
            return 0.0
        min_dist = min(
            abs(x - hx) + abs(y - hy) for hx, hy in hazard_cells
        )
        return 1.0 / (1.0 + min_dist)

    def calculate_expected_utility(self, target_x: int, target_y: int,
                                    env_state, weights: list) -> float:
        """
        E[U] = w · sign · x   where sign = [+1, +1, -1, +1, -1]

        Positive weights reward exploration and survivor probability;
        negative signs on distance and hazard penalty make the agent prefer
        nearby, safe moves.  The GA evolves the magnitudes; signs are fixed
        by domain semantics.
        """
        features = self.get_decision_features(target_x, target_y, env_state)
        w = np.array(weights[:FEATURE_DIM], dtype=float)
        signs = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        return float(np.dot(w * signs, features))

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def find_path_to_nearest_frontier(self, env_state, weights: list) -> list:
        """
        BFS to the nearest frontier tier; expected utility breaks ties within
        the same BFS distance.

        Uses a back-pointer dict instead of copying path lists into the queue,
        reducing memory from O(V * avg_path) to O(V).
        """
        start = (self.x, self.y)
        # parent[node] = node that reached it (None for start)
        parent: dict[tuple[int,int], tuple[int,int] | None] = {start: None}
        depth: dict[tuple[int,int], int] = {start: 0}
        queue: deque[tuple[int,int]] = deque([start])

        frontier_nodes: list[tuple[int,int]] = []
        cutoff_depth: int | None = None

        while queue:
            node = queue.popleft()
            d = depth[node]

            if cutoff_depth is not None and d > cutoff_depth:
                break

            if self._is_frontier(*node, env_state):
                cutoff_depth = d
                frontier_nodes.append(node)
                # Don't expand beyond a frontier — we already have it
                continue

            for nb in self._neighbors(*node):
                if nb in parent:
                    continue
                if not self._is_known_safe_to_travel(*nb, env_state):
                    continue
                parent[nb] = node
                depth[nb] = d + 1
                queue.append(nb)

        if not frontier_nodes:
            return []

        # Pick the frontier with the highest expected utility score
        best = max(
            frontier_nodes,
            key=lambda n: self._frontier_score(n, env_state, weights),
        )

        # Reconstruct path from start → best
        path: list[tuple[int,int]] = []
        cur = best
        while cur != start:
            path.append(cur)
            cur = parent[cur]  # type: ignore[assignment]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbors(self, x: int, y: int) -> list[tuple[int,int]]:
        return [
            (x + dx, y + dy)
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]
            if 0 <= x + dx < self.env_width and 0 <= y + dy < self.env_height
        ]

    def _sensor_cells(self, x: int, y: int) -> list[tuple[int,int]]:
        return [
            (x + dx, y + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if 0 <= x + dx < self.env_width and 0 <= y + dy < self.env_height
        ]

    def _is_frontier(self, x: int, y: int, env_state=None) -> bool:
        if env_state is None:
            return False
        if not self._is_known_safe_to_travel(x, y, env_state):
            return False
        return any(
            self.internal_map[nx, ny] == -1
            for nx, ny in self._sensor_cells(x, y)
        )

    def _is_known_safe_to_travel(self, x: int, y: int, env_state) -> bool:
        return (
            self.internal_map[x, y] != -1
            and env_state.is_traversable(x, y)
            and self.kb.ask(f"Safe_{x}_{y}")
        )

    def _frontier_score(self, cell: tuple[int,int], env_state, weights: list) -> float:
        """Score used to break ties among equidistant frontiers."""
        return self.calculate_expected_utility(cell[0], cell[1], env_state, weights)
