import genetic_algorithm as ga

# Specify parameters
name = "75CRate"
generations = 300
tables = 25  # Multiplied by 8, population num
games = 5
elites = 2
c_rate = 0.75
m_rate = 0.1
tournaments = 10  # Size of tournaments in selection

# Run GA
new_ga = ga.GeneticAlgorithm(generations, tables, games, elites, c_rate, m_rate, tournaments)
result = new_ga.run()

# Save to file - using a dictionary
agent_dict = {
    'Name': name,
    'Generations' : generations,
    'Tables' : tables,
    'Games' : games,
    'Elites ' : elites,
    'Crossover Rate ' : c_rate,
    'Mutation Rate ' : m_rate,
    'Tournament Size ' : tournaments,
    'Weights': ", ".join(str(w) for w in result.m_weights),
    'Biases': ", ".join(str(b) for b in result.m_biases)
}

# Write to file in the BestAgents folder
with open(str("BestAgents\\" + name + ".txt"),'w') as f:
    for key, value in agent_dict.items():
        f.write('%s:%s\n' % (key, value))

    f.close()

print("Run complete.")
