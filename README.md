# evolving-poker-agents
Genetic algorithm trained neural-network architecture to play Poker.  

The project was part of my master's degree dissertation. The aim was to implement a neural network architecture that's able to interface with an existing poker implementation and play poker. The agent takes information from the game state and makes decisions through running the network and therefore plays the game. The project uses a genetic algorithm in order to train the neural networks. Basically a bunch of neural networks play poker against each-other, and the best are evaluated and selected to be part of a new generation, where new poker agents are generated. This keeps going until we've found the ultimate poker agent.  

I think for the most part the project was successful, some of the AI elements fell short but I'm satisfied with what I made.  

Poker simulator: PyPokerEngine https://github.com/ishikota/PyPokerEngine by ishikota  
Poker simulator with a few fixes: PyPokerEngine https://github.com/schreven/PyPokerEngine by schreven  
Poker hand evaluator: Treys https://github.com/ihendley/treys by ihendley  
