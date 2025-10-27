import random
import numpy as np
import matplotlib.pyplot as plt

def booth_function(x: float, y: float) -> float:
    """
    Booth function in 2D.
    Global minimum is at (1, 3), with f(1, 3) = 0.
    Range of interest: x, y ∈ [-5,5].
    """
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

class GeneticAlgorithm:
    """
    Genetic Algorithm implementation for minimizing the Booth function.

    Using:
    - Roulette Wheel Selection
    - Random Interpolation Crossover
    - Gaussian Mutation
    and a fitness = log(1 + 1 / cost).
    """
    def __init__(self,
                 population_size=100,
                 mutation_rate=0.2,
                 mutation_strength=0.05,
                 crossover_rate=0.8,
                 num_generations=100):
        """
        Initialize GA parameters and function bounds.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        
        # For the Booth function:
        self.fn = booth_function
        self.x_bounds = (-5, 5)
        self.y_bounds = (-5, 5)

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)

    def initialize_population(self) -> np.ndarray:
        """
        Create an initial population of (population_size x 2),
        each row representing (x, y).
        """
        pop = []
        for _ in range(self.population_size):
            x = random.uniform(self.x_bounds[0], self.x_bounds[1])
            y = random.uniform(self.y_bounds[0], self.y_bounds[1])
            pop.append((x, y))
        return np.array(pop)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Compute the 'fitness' of each individual.
        
        NEW FORMULA:
          fitness = log(1 + 1 / cost),
        so smaller cost => bigger fitness.

        We'll add a small epsilon to avoid division by zero
        if cost is extremely small.
        """
        fitness_values = []
        for (x, y) in population:
            cost = self.fn(x, y)
            # Avoid dividing by zero if cost ~ 0
            if cost < 1e-12:
                cost = 1e-12
            # log(1 + 1/cost)
            fitness = np.log(1.0 + 1.0 / cost)
            fitness_values.append(fitness)
        return np.array(fitness_values)

    def selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """
        Roulette Wheel Selection:
        - Convert fitness array to probabilities.
        - Use random draws to select 'parents' for reproduction.
        """
        new_population = []
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            # If total_fitness is zero (edge case), pick uniformly
            probabilities = np.ones_like(fitnesses) / len(fitnesses)
        else:
            probabilities = fitnesses / total_fitness

        for _ in range(self.population_size):
            r = random.random()
            cumulative = 0.0
            for i, p in enumerate(probabilities):
                cumulative += p
                if cumulative >= r:
                    new_population.append(population[i])
                    break
        return np.array(new_population)

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Random Interpolation Crossover:
          - For pairs of parents, produce 2 offspring via interpolation.
          - We only do crossover for a fraction (crossover_rate) of the pairs.
        """
        offspring = parents.copy()
        
        # Number of pairs on which to perform crossover
        num_pairs = int(self.crossover_rate * (len(parents) // 2))
        indices = np.arange(len(parents))
        np.random.shuffle(indices)

        for i in range(num_pairs):
            idx1 = indices[2*i]
            idx2 = indices[2*i + 1]
            p1 = parents[idx1]
            p2 = parents[idx2]

            alpha = random.random()  # interpolation factor in [0,1]

            # Child 1
            c1 = alpha * p1 + (1.0 - alpha) * p2
            # Child 2
            c2 = alpha * p2 + (1.0 - alpha) * p1

            offspring[idx1] = c1
            offspring[idx2] = c2

        return offspring

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """
        Gaussian Mutation:
          - With probability mutation_rate, add N(0, mutation_strength) to x and y.
          - Clip values to [x_bounds, y_bounds].
        """
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                dx = random.gauss(0, self.mutation_strength)
                dy = random.gauss(0, self.mutation_strength)
                new_x = population[i][0] + dx
                new_y = population[i][1] + dy
                # Clip to [-5, 5]
                new_x = max(self.x_bounds[0], min(self.x_bounds[1], new_x))
                new_y = max(self.y_bounds[0], min(self.y_bounds[1], new_y))
                population[i] = (new_x, new_y)
        return population

    def evolve(self, seed=42):
        """
        Main GA loop:
          - Initialize population
          - Evaluate & record best/average fitness each generation
          - Selection -> Crossover -> Mutation -> Next generation
        Returns:
          - best_solutions: list of best individuals each generation
          - best_fits: list of best fitness each generation
          - avg_fits: list of average fitness each generation
        """
        self.set_seed(seed)
        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        avg_fitness_values = []

        for _ in range(self.num_generations):
            # Evaluate current population
            fitness_values = self.evaluate_population(population)

            # Find best individual in this generation
            best_index = np.argmax(fitness_values)
            best_fit = fitness_values[best_index]
            avg_fit = np.mean(fitness_values)

            best_solutions.append(population[best_index].copy())
            best_fitness_values.append(best_fit)
            avg_fitness_values.append(avg_fit)

            # Selection
            selected = self.selection(population, fitness_values)
            # Crossover
            crossed = self.crossover(selected)
            # Mutation
            population = self.mutate(crossed)

        return best_solutions, best_fitness_values, avg_fitness_values

###############################################################################
#                                 MAIN DEMO                                   #
###############################################################################
if __name__ == "__main__":
    # Instantiate the GA with desired parameters
    ga = GeneticAlgorithm(
        population_size=200,
        mutation_rate=1.0,
        mutation_strength=2.0,
        crossover_rate=0.9,
        num_generations=100
    )

    # Run the GA for a single seed
    best_solutions, best_fits, avg_fits = ga.evolve(seed=2)

    # Final solution and cost
    final_best_sol = best_solutions[-1]
    final_cost = booth_function(final_best_sol[0], final_best_sol[1])

    # We already have best fitness from best_fits[-1]:
    final_fitness = best_fits[-1]

    print("Best solution (x, y) found:", final_best_sol)
    print("Booth function value at best solution:", final_cost)
    print("Best fitness at that solution (log-based):", final_fitness)

    # Plot Best vs. Average Fitness over generations
    generations = np.arange(len(best_fits))
    plt.figure()
    plt.plot(generations, best_fits, label="Best Fitness")
    plt.plot(generations, avg_fits, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness = log(1 + 1/cost)")
    plt.title("GA Convergence - Booth Function (log-based fitness)")
    plt.legend()
    plt.show()
