from collections import deque
import numpy as np
from logic.knowledge_base import KnowledgeBase # IMPORT THE NEW KB

class DroneState:
    def __init__(self, start_x: int, start_y: int, max_battery: int, env_width: int, env_height: int):
        self.x = start_x
        self.y = start_y
        self.battery = max_battery
        self.is_active = True
        
        self.env_width = env_width
        self.env_height = env_height
        self.internal_map = np.full((env_width, env_height), -1, dtype=int)
        
        # --- PILLAR 1: LOGIC INTEGRATION ---
        self.kb = KnowledgeBase()
        self.rescue_beacons = {}
        
        # The drone knows its starting point is safe
        self.mark_safe(self.x, self.y)

    def mark_safe(self, x: int, y: int):
        """Helper to update both the internal map and the Logic KB."""
        if 0 <= x < self.env_width and 0 <= y < self.env_height:
            self.internal_map[x, y] = 0
            self.kb.add_fact(f"Safe_{x}_{y}")

    def sense_environment(self, actual_env_state):
        """
        Simulates the drone's sensors. Includes a camera that maps the immediate 
        3x3 area, plus the logical inference for hidden dangers.
        """
        changed_cells = []

        # 1. Camera Sensor: Reveal the immediate 3x3 area around the drone
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.env_width and 0 <= ny < self.env_height:
                    # Look at the real world and copy it to the drone's internal map
                    actual_val = actual_env_state.get_cell_truth(nx, ny)
                    if self.internal_map[nx, ny] != actual_val:
                        changed_cells.append((nx, ny))
                    self.internal_map[nx, ny] = actual_val
                    if actual_env_state.has_survivor(nx, ny) and (nx, ny) not in self.rescue_beacons:
                        confidence = actual_env_state.detection_confidence(nx, ny)
                        self.rescue_beacons[(nx, ny)] = confidence
                        if confidence < 1.0:
                            print(f"Possible survivor under tree cover at ({nx}, {ny})! Calling for help.")
                        else:
                            print(f"Survivor found at ({nx}, {ny})! Calling for help.")

        # 2. Logic Engine (Keep your existing logic for the cell the drone is ON)
        current_val = actual_env_state.get_cell_truth(self.x, self.y)
        neighbors = [
            (self.x + 1, self.y), (self.x - 1, self.y),
            (self.x, self.y + 1), (self.x, self.y - 1)
        ]
        
        if actual_env_state.is_traversable(self.x, self.y):
            self.kb.add_fact(f"Clear_{self.x}_{self.y}")
            for nx, ny in neighbors:
                if 0 <= nx < self.env_width and 0 <= ny < self.env_height:
                    self.kb.add_rule([f"Clear_{self.x}_{self.y}"], f"Safe_{nx}_{ny}")
                    
        elif current_val == 1:
            self.kb.add_fact(f"Blocked_{self.x}_{self.y}")
            
        # Run inference
        self.kb.infer()
        return changed_cells

    def move_to(self, target_x: int, target_y: int):
        """Move only when propositional logic proves the target is safe."""
        if self.kb.ask(f"Safe_{target_x}_{target_y}"):
            self.x = target_x
            self.y = target_y
            return True
        else:
            print(f"Logic Error: Drone refuses to move to ({target_x}, {target_y}). It is not proven Safe!")
            return False

    def move(self, dx: int, dy: int):
        return self.move_to(self.x + dx, self.y + dy)

    def get_decision_features(self, target_x: int, target_y: int, env_state) -> np.ndarray:
        """
        PILLAR 2: THE MATH OF AI
        Returns x = [unexplored, P(survivor), normalized_distance].
        This explicitly models the state as a linear algebra feature vector with
        probability as one dimension.
        """
        center_vector = np.array([self.env_width / 2, self.env_height / 2])
        target_vector = np.array([target_x, target_y])
        distance_norm = np.linalg.norm(center_vector - target_vector)
        max_distance = np.linalg.norm(center_vector)
        normalized_distance = distance_norm / max_distance if max_distance else 0.0

        if self.internal_map[target_x, target_y] == -1:
            survivor_prob = env_state.survivor_probabilities[target_x, target_y]
            is_unexplored = 1.0
        else:
            survivor_prob = 0.0
            is_unexplored = 0.0

        return np.array([is_unexplored, survivor_prob, normalized_distance])

    def calculate_expected_utility(self, target_x: int, target_y: int, env_state, weights: list) -> float:
        """
        Calculates expected utility using a dot product:
        E[U] = w_explore*x_explore + w_prob*P(survivor) - w_dist*distance.
        """
        features = self.get_decision_features(target_x, target_y, env_state)
        signed_weights = np.array([weights[0], weights[1], -weights[2]])
        return float(np.dot(signed_weights, features))

    def find_path_to_nearest_frontier(self, env_state, weights: list) -> list:
        """
        Finds the shortest path to the nearest frontier cell using nearest-neighbor
        coverage. A frontier is a known-safe travel cell whose 3x3 sensor footprint
        still contains unchecked territory. BFS makes distance the first priority;
        expected utility only breaks ties between equally near frontier cells.
        """
        start = (self.x, self.y)
        queue = deque([(start, [])])
        visited = {start}
        frontier_paths = []
        current_distance = None

        while queue:
            (x, y), path = queue.popleft()

            if current_distance is not None and len(path) > current_distance:
                break

            if self._is_frontier(x, y, env_state):
                current_distance = len(path)
                frontier_paths.append(path)
                continue

            for nx, ny in self._neighbors(x, y):
                if (nx, ny) in visited:
                    continue
                if not self._is_known_safe_to_travel(nx, ny, env_state):
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

        if not frontier_paths:
            return []

        return max(
            frontier_paths,
            key=lambda path: self._frontier_score(path[-1], env_state, weights)
            if path else self._frontier_score(start, env_state, weights)
        )

    def _neighbors(self, x: int, y: int) -> list:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [
            (nx, ny)
            for nx, ny in candidates
            if 0 <= nx < self.env_width and 0 <= ny < self.env_height
        ]

    def _sensor_cells(self, x: int, y: int) -> list:
        cells = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env_width and 0 <= ny < self.env_height:
                    cells.append((nx, ny))
        return cells

    def _is_frontier(self, x: int, y: int, env_state=None) -> bool:
        if env_state is None:
            return False
        if not self._is_known_safe_to_travel(x, y, env_state):
            return False

        return any(self.internal_map[nx, ny] == -1 for nx, ny in self._sensor_cells(x, y))

    def _is_known_safe_to_travel(self, x: int, y: int, env_state) -> bool:
        return (
            self.internal_map[x, y] != -1
            and env_state.is_traversable(x, y)
            and self.kb.ask(f"Safe_{x}_{y}")
        )

    def _frontier_score(self, cell: tuple, env_state, weights: list) -> float:
        """
        Tie-breaks equally near frontier cells with expected value and coverage.
        The primary optimization remains shortest distance from BFS.
        """
        x, y = cell
        unchecked_sensor_cells = sum(
            1 for nx, ny in self._sensor_cells(x, y)
            if self.internal_map[nx, ny] == -1
        )
        return (
            self.calculate_expected_utility(x, y, env_state, weights)
            + unchecked_sensor_cells
        )
