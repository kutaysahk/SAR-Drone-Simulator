import random
import numpy as np


class EnvironmentState:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=int)
        self.survivor_probabilities = np.full((width, height), 0.05)
        self.survivor_locations = set()
        self.mode = "simulation"

    def add_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x, y] = 1

    def add_hazard(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x, y] = 2

    def add_survivor(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x, y] = 3
            self.survivor_locations.add((x, y))
            self.survivor_probabilities[x, y] = 1.0

    def get_cell_truth(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x, y]
        return 1

    def generate_disaster_zone(self, obstacle_density: int, hazard_density: int, num_survivors: int):
        """Randomly generates a disaster map with rubble, fire, and hidden survivors."""
        self.grid[:, :] = 0
        self.survivor_probabilities[:, :] = 0.05
        self.survivor_locations = set()

        for _ in range(obstacle_density):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.add_obstacle(x, y)
            if random.random() > 0.3:
                self.add_obstacle(x + 1, y)
            if random.random() > 0.3:
                self.add_obstacle(x, y + 1)
            if random.random() > 0.6:
                self.add_obstacle(x + 2, y)

        for _ in range(hazard_density):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if self.grid[x, y] == 0:
                self.add_hazard(x, y)
                if random.random() > 0.5:
                    self.add_hazard(x - 1, y)
                if random.random() > 0.5:
                    self.add_hazard(x, y - 1)

        self.grid[0:3, 0:3] = 0
        self.survivor_probabilities[0:3, 0:3] = 0.0

        survivor_candidates = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[x, y] == 0 and not (x < 3 and y < 3)
        ]
        random.shuffle(survivor_candidates)
        for x, y in survivor_candidates[:num_survivors]:
            self.add_survivor(x, y)

    def has_survivor(self, x: int, y: int) -> bool:
        return (x, y) in self.survivor_locations

    def detection_confidence(self, x: int, y: int) -> float:
        return 1.0

    def is_traversable(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.grid[x, y] in [0, 3]
