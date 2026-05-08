import random
import numpy as np

class GeneticOptimizer:
    """
    PILLAR 3: OPTIMIZATION (Genetic Algorithm)
    Evolves the decision-making weights for the drone's Expected Utility function.
    """
    def __init__(self, population_size=20, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        # A chromosome is [Weight_Explore, Weight_Probability, Weight_Distance]
        self.population = [self._random_chromosome() for _ in range(population_size)]

    def _random_chromosome(self):
        """Generates bounded weights for [exploration, survivor probability, travel cost]."""
        return [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]

    def fitness_function(self, chromosome):
        """
        Scores weights on representative rescue states.
        The chromosome is a linear algebra weight vector w, each scenario is a
        feature vector x = [unexplored, survivor_probability, normalized_distance],
        and utility is the dot product w.x.
        """
        weights = np.array(chromosome)

        positive_cases = np.array([
            [1.0, 0.80, 0.15],
            [1.0, 0.55, 0.25],
            [0.6, 0.35, 0.10],
        ])
        negative_cases = np.array([
            [0.0, 0.00, 0.10],
            [0.2, 0.05, 0.80],
            [0.0, 0.10, 0.65],
        ])

        reward = float(np.mean(positive_cases @ weights))
        penalty = float(np.mean(negative_cases @ weights))
        distance_discipline = 1.0 - abs(chromosome[2] - 0.35)
        return reward - penalty + distance_discipline

    def selection(self):
        """Selects the top 50% of the population based on fitness."""
        scored_population = [(self.fitness_function(chrom), chrom) for chrom in self.population]
        # Sort descending by score
        scored_population.sort(key=lambda x: x[0], reverse=True) 
        
        # Keep the top half
        half_length = len(self.population) // 2
        return [chrom for score, chrom in scored_population[:half_length]]

    def crossover_and_mutate(self, parents):
        """Creates a new generation through crossover and mutation."""
        new_population = []
        
        while len(new_population) < self.population_size:
            # Pick two random parents from the elite group
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover: Mix their genes
            child = [
                parent1[0] if random.random() > 0.5 else parent2[0],
                parent1[1] if random.random() > 0.5 else parent2[1],
                parent1[2] if random.random() > 0.5 else parent2[2]
            ]
            
            # Mutation: Randomly alter a gene
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    child[i] += random.uniform(-0.15, 0.15) # Shift weight slightly
                    child[i] = min(1.0, max(0.0, child[i])) # Keep it bounded
                    
            new_population.append(child)
            
        self.population = new_population

    def run_evolution(self, generations=50):
        """Runs the genetic algorithm and returns the best chromosome."""
        print("--- Running Genetic Algorithm Optimization ---")
        for g in range(generations):
            elites = self.selection()
            self.crossover_and_mutate(elites)
            
        # Return the best chromosome of the final generation
        best_chromosome = max(self.population, key=self.fitness_function)
        print(f"Optimization Complete. Best Weights: {best_chromosome}")
        return best_chromosome
