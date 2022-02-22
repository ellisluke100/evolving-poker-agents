from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker
from neural_net_player import NeuralNetPlayer
import random
import time
import numpy as np
from treys import Evaluator
import heapq

class GeneticAlgorithm:

    # Adjustable parameters
    TABLE_SIZE = 8
    POPULATION_SIZE = TABLE_SIZE * 4
    ELITE_NUM = 2
    TOURNAMENT_SIZE = 3

    GENERATION_NUM = 30
    RUN_COUNT = 1
    GAME_NUM = 20

    M_RATE = 0.1
    C_RATE = 0.9

    m_population = list()
    m_scores = list()
    times = list()

    # For use in hand evaluation, to send to each neural network
    hand_evaluator = Evaluator()

    def __init__(self, gen, tables, games, elites, c_rate, m_rate, t_size):

        # Set the GA parameters from what we are sent in
        self.GENERATION_NUM = gen
        self.POPULATION_SIZE = tables * self.TABLE_SIZE
        self.GAME_NUM = games
        self.ELITE_NUM = elites
        self.C_RATE = c_rate
        self.M_RATE = m_rate
        self.TOURNAMENT_SIZE = t_size

    def population_initialization(self):
        # Initialize each player in the initial population
        for i in range(0, self.POPULATION_SIZE):
            self.m_population.append(NeuralNetPlayer("p"+str(i), self.hand_evaluator, True))

    def evaluate_fitness(self):
        self.m_scores.clear()

        # Evaluate in chunks of TABLE_SIZE, i.e. by playing games
        for i in range(0, self.POPULATION_SIZE, self.TABLE_SIZE):
            config = setup_config(max_round=10, initial_stack=300, small_blind_amount=10)

            # Add players to config
            for j in range(0, self.TABLE_SIZE):
                config.register_player(name=self.m_population[i+j].name, algorithm=self.m_population[i + j])
                self.m_population[i+j].set_uuid("uuid"+str(j+1))

            # Play X games
            for x in range(0, self.GAME_NUM):
                start_poker(config, verbose=0)

            # Calculate fitnesses and store in the scores list
            for y in range(0, self.TABLE_SIZE):
                self.m_scores.append(self.m_population[i+y].calculate_fitness(self.GAME_NUM))
                self.m_population[i+y].in_same_game_set = False

    def create_new_population(self):
        new_population = list()

        # Keep the ELITE_NUM best individuals
        elites = heapq.nlargest(self.ELITE_NUM, self.m_scores)
        for e in elites:
            ind = self.m_scores.index(e)
            new_population.append(self.m_population[ind])

        for i in range(self.ELITE_NUM, self.POPULATION_SIZE):
            new_individual = (self.crossover(self.tournament_selection(), self.tournament_selection()))
            new_population.append(new_individual)

    def tournament_selection(self):
        best_individual = -1

        for k in range(0, self.TOURNAMENT_SIZE):
            random_individual = random.randint(0, self.POPULATION_SIZE-1)
            if best_individual == -1 or self.m_scores[random_individual] > self.m_scores[best_individual]:
                best_individual = random_individual

        return self.m_population[best_individual]

    def crossover(self, parent1, parent2):
        # Random weighted sum crossover
        # Generate a random weight, sum the parent weights, multiply by weight, set as child weight
        child = parent1

        val = random.uniform(0, 1)
        if val < self.C_RATE:
            for i in range(0, child.get_num_weights()):
                sum_weight1 = random.uniform(0, 1)
                child.m_weights[i] = sum_weight1 * (parent1.m_weights[i] + parent2.m_weights[i])

            for j in range(0, child.get_num_biases()):
                sum_bias1 = random.uniform(0, 1)
                child.m_biases[j] = sum_bias1 * (parent1.m_biases[j] + parent2.m_biases[j])

        return child

    def mutation(self):
        # Additive weight and bias mutation
        # For each weight, then each bias, in each member of the population
        # If mutation rate = true, select random weight / bias from initial distribution and add it to the allele

        for i in range(0, self.POPULATION_SIZE):
            val = 0
            individual = self.m_population[i]
            for j in range(0, individual.get_num_weights()):
                val = random.uniform(0, 1)
                if val < self.M_RATE:
                    add_value = random.uniform(-0.5, 0.5)
                    individual.set_new_weight(add_value, j)

            for y in range(0, individual.get_num_biases()):
                val = random.uniform(0, 1)
                if val < self.M_RATE:
                    add_value = random.uniform(0.5, 0.5)
                    individual.set_new_bias(add_value, y)

    def run(self):
        start_time = time.time()
        best_fits_per_gen = list()

        self.population_initialization()  # Randomly initialize the first population
        print("Initialization complete.")
        self.evaluate_fitness()

        # For each generation, perform selection, crossover, mutation, and fitness evaluation
        for i in range(0, self.GENERATION_NUM):
            t = time.time()
            self.create_new_population()
            self.mutation()
            random.shuffle(self.m_population) # Shuffle so agents aren't just playing games against the same 7 others

            self.evaluate_fitness()

            print("Generation ", i+1, " complete. Best: ", max(self.m_scores), "Time taken: ", time.time() - t, "secs")
            best_fits_per_gen.append(max(self.m_scores))
            self.times.append(time.time()-t)

        print("Time taken: ", time.time()-start_time, "secs")
        print("Average time per gen: ", np.average(self.times))

        print("Finished.")
        print("Best individual: ", max(self.m_scores))

        # Return the best individual
        return self.m_population[self.m_scores.index(max(self.m_scores))]

