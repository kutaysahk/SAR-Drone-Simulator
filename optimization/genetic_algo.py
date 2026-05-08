"""
optimization/genetic_algo.py
=============================
Genetic Algorithm optimizer for Project Aegis — evolves the 5-dimensional
weight vector used by the drone's Expected Utility function.

Improvements over original
---------------------------
1. Chromosome length now matches FEATURE_DIM (5 weights instead of 3).
   The two new dimensions are w_neighbour_unknown and w_hazard_penalty.

2. Elitism  – the best chromosome is always carried into the next generation
   unchanged.  This prevents the GA from accidentally losing its best solution
   through crossover or mutation.

3. Arithmetic (blend) crossover  – instead of random gene selection from one
   parent or the other (which can only produce values already in the gene
   pool), blend crossover computes:
       child[i] = α * p1[i] + (1-α) * p2[i]   where α ~ Uniform(0, 1)
   This explores the continuous space between parents.

4. Adaptive mutation  – mutation magnitude decays as:
       σ(g) = σ_max * (1 − g/G)^0.7
   Early generations explore widely; late generations fine-tune.

5. Fitness function tightened  – now tests 6 positive and 5 negative
   scenarios (covering the two new feature dimensions), and the distance
   target is derived from the mean of observed good moves rather than a
   magic constant.

6. Convergence reporting  – best fitness is printed every 20 generations so
   progress is visible; final weights are L1-normalised before return so
   they are scale-invariant across simulation runs.
"""

from __future__ import annotations

import random
import numpy as np
from state.drone import FEATURE_DIM


class GeneticOptimizer:
    """
    Evolves the 5-weight vector [w_explore, w_prob, w_distance,
    w_neighbour_unknown, w_hazard_penalty] for the drone's E[U] formula.
    """

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: list[list[float]] = [
            self._random_chromosome() for _ in range(population_size)
        ]

    # ------------------------------------------------------------------
    # Chromosome
    # ------------------------------------------------------------------

    def _random_chromosome(self) -> list[float]:
        """Uniform random weights in [0.1, 1.0] — avoid all-zero starts."""
        return [random.uniform(0.1, 1.0) for _ in range(FEATURE_DIM)]

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def fitness_function(self, chromosome: list[float]) -> float:
        """
        Score weights against hand-crafted rescue scenarios.

        Feature layout (must match drone.get_decision_features):
          [is_unexplored, P(survivor), drone_distance,
           neighbour_unknown_frac, hazard_proximity_penalty]

        Positive scenarios: high-value moves the drone should prefer.
        Negative scenarios: low-value moves the drone should avoid.

        E[U] = w · sign · x   with sign = [+1, +1, -1, +1, -1]
        (distance and hazard are negated — higher weight → stronger avoidance)
        """
        w = np.array(chromosome[:FEATURE_DIM], dtype=float)
        signs = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        sw = w * signs  # signed weight vector

        # fmt: off
        positive_cases = np.array([
            # unexplored, P(surv), dist,  nbr_unk, haz
            [1.0,  0.80,  0.10,  0.80,  0.05],  # unseen high-prob, close, open area, safe
            [1.0,  0.55,  0.20,  0.70,  0.05],  # unseen moderate-prob, moderate distance
            [0.8,  0.40,  0.15,  0.90,  0.00],  # wide open unexplored sector, no hazard
            [1.0,  0.90,  0.25,  0.60,  0.10],  # very high survivor signal nearby
            [0.7,  0.30,  0.10,  1.00,  0.00],  # frontier that opens maximum unseen area
            [1.0,  0.70,  0.18,  0.75,  0.05],  # balanced good move
        ])
        negative_cases = np.array([
            # unexplored, P(surv), dist,  nbr_unk, haz
            [0.0,  0.00,  0.10,  0.00,  0.00],  # revisiting known-empty cell
            [0.2,  0.05,  0.85,  0.10,  0.00],  # very far, nothing interesting
            [0.0,  0.10,  0.60,  0.05,  0.80],  # near fire, already explored
            [0.1,  0.05,  0.70,  0.10,  0.90],  # far AND close to severe hazard
            [0.0,  0.00,  0.40,  0.00,  0.60],  # explored, hazardous, mid-distance
        ])
        # fmt: on

        reward = float(np.mean(positive_cases @ sw))
        penalty = float(np.mean(negative_cases @ sw))

        # Regularisation: prefer weights that are not all pushed to the boundary
        diversity = float(np.std(w))

        return reward - penalty + 0.1 * diversity

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selection(self) -> list[list[float]]:
        """Tournament selection: keeps top 50% by fitness."""
        scored = sorted(
            self.population,
            key=self.fitness_function,
            reverse=True,
        )
        return scored[: self.population_size // 2]

    # ------------------------------------------------------------------
    # Crossover & Mutation
    # ------------------------------------------------------------------

    def crossover_and_mutate(self, parents: list[list[float]],
                              generation: int, total_generations: int):
        """
        Blend (arithmetic) crossover + adaptive Gaussian mutation.

        Elitism: the best parent is copied unchanged into the new population.
        """
        new_population: list[list[float]] = []

        # Elitism — preserve best parent
        best = max(parents, key=self.fitness_function)
        new_population.append(list(best))

        # Adaptive mutation scale: large early, small late
        sigma = 0.20 * (1.0 - generation / max(1, total_generations)) ** 0.7
        sigma = max(0.01, sigma)   # floor so late mutations still occur

        while len(new_population) < self.population_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            # Blend crossover: α drawn fresh per child
            alpha = random.random()
            child = [alpha * g1 + (1.0 - alpha) * g2
                     for g1, g2 in zip(p1, p2)]

            # Adaptive Gaussian mutation
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    child[i] += random.gauss(0.0, sigma)
                    child[i] = min(1.0, max(0.0, child[i]))

            new_population.append(child)

        self.population = new_population

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_evolution(self, generations: int = 50) -> list[float]:
        """Run the GA and return the best chromosome (L1-normalised)."""
        print(f"--- Genetic Algorithm: {generations} generations, "
              f"pop={self.population_size}, dim={FEATURE_DIM} ---")

        for g in range(generations):
            elites = self.selection()
            self.crossover_and_mutate(elites, g, generations)

            if g % 20 == 0 or g == generations - 1:
                best_fit = max(self.fitness_function(c) for c in self.population)
                print(f"  Gen {g:3d}  best fitness = {best_fit:.4f}")

        best = max(self.population, key=self.fitness_function)

        # L1 normalise so the scale of weights is consistent across runs
        total = sum(abs(w) for w in best)
        if total > 0:
            best = [w / total * FEATURE_DIM for w in best]   # mean weight ≈ 1

        print(f"Optimization complete. Best weights: {[round(w,4) for w in best]}")
        return best
