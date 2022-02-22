from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker
from neural_net_player import NeuralNetPlayer
from treys import Evaluator
import matplotlib.pyplot as plt
import numpy as np

GAME_NUM = 500

evaluator = Evaluator()

# Define the agents we want
agent_paths = ["065CRate.txt",
          "07CRate.txt",
          "075CRate.txt",
          "Control.txt",
          "085CRate.txt",
          "09CRate.txt",
          "095CRate.txt",
          "1CRate.txt"
          ]

agents = list()
i = 0

# Load agents
for path in agent_paths:
    i += 1
    d = {}

    # Open file and read each line into a dictionary
    with open("BestAgents/"+path) as f:
        for line in f:
            (key, val) = line.split(':')
            val = val.strip('\n')
            d[key] = val

        f.close()

        new_player = NeuralNetPlayer(d['Name']+str(i), evaluator, True)

        w_string = ""
        b_string = ""

        # Set the agent's weights and biases to the ones we read from the file
        weights = d['Weights']
        for w in weights:
            if w != ',':
                w_string = w_string + w
            elif w != ' ':
                new_player.m_weights.append(float(w_string))
                w_string = ""

        biases = d['Biases']
        for b in biases:
            if b != ',':
                b_string = b_string + b
            elif b != ' ':
                new_player.m_biases.append(float(b_string))
                b_string = ""

        agents.append(new_player)

results_dict = {}

config = setup_config(max_round=10, initial_stack=300, small_blind_amount=10)

for agent in agents:
    config.register_player(agent.name, agent)

# Play games
for i in range(0, GAME_NUM):
    start_poker(config, verbose=0)
    print("Game "+str(i+1)+" finished.")

# Get fitnesses for each agent
for agent in agents:
    fitness = agent.calculate_fitness(GAME_NUM)

    print("Agent: " + agent.name + " , Score: " + str(fitness))
    results_dict[agent.name] = fitness

results_list = list()
for v in results_dict:
    results_list.append(results_dict[v])

agent_names = ["65%",
               "70%",
               "75%",
               "80%",
               "85%",
               "90%",
               "95%",
               "100%"]

x_pos = [i for i, _ in enumerate(agent_names)]

plt.bar(x_pos, results_list, color='blue')
plt.xlabel("Crossover rate")
plt.ylabel("Average chip count per game")
plt.title("Barchart comparing effect of increasing crossover rate")

plt.xticks(x_pos, agent_names)
plt.show()






