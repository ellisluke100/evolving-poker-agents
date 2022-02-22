from PyPokerEngine.pypokerengine.players import BasePokerPlayer
import numpy as np
import random
import math
from treys import Card

# General class description

class NeuralNetPlayer(BasePokerPlayer):
    # Neural Network

    # Structural Parameters
    NUM_HIDDEN_LAYERS = 2
    NUM_HIDDEN_NODES = 7
    NUM_OUTPUTS = 5
    NUM_INPUTS = 19
    TOTAL_CONNECTIONS = 0

    def __init__(self, id, evaluator, training):
        super().__init__()

        self.rankings = list()
        self.training = training

        self.hand_evaluator = evaluator
        self.current_hand_strength = 0
        self.current_hole_strength = 0
        self.current_hole = []

        # Network Structure
        self.m_layers = list()
        self.m_weights = list()
        self.m_biases = list()
        self.TOTAL_CONNECTIONS = self.calculate_total_connections()
        self.NUM_BIASES = self.NUM_HIDDEN_LAYERS*self.NUM_HIDDEN_NODES + self.NUM_OUTPUTS

        # uuid = 'uuid-'+str(player_nb)
        self.name = id  # Name is what is referred to in the bigger scope
        self.uuid = 0  # ID referred to in a table

        # Inputs into the network
        self.chip_count = 0  # Self chip count
        self.chip_counts = list()  # List of chip counts per player
        self.chips_in_pot = 0
        self.num_players_left = 0

        self.card_values = {
            "A": 10,
            "K": 8,
            "Q": 7,
            "J": 6,
            "T": 5,
            "9": 4.5,
            "8": 4,
            "7": 3.5,
            "6": 3,
            "5": 2.5,
            "4": 2,
            "3": 1.5,
            "2": 1
        }
        self.card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

        self.opponent_aggressions = {}
        self.in_same_game_set = False

        # For use in fitness evaluation
        self.total_chip_count = 0

        # First time initialisation of layers
        for i in range(0, self.NUM_HIDDEN_LAYERS + 2):
            self.m_layers.append(list())

        self.initialize_network()

    def get_chip_count(self):
        return self.chip_count

    def init_weights(self):
        # Randomly initialize weights, between -1 and 1
        for i in range(0, self.TOTAL_CONNECTIONS):
            self.m_weights.append(random.uniform(-1, 1))

    def init_biases(self):
        # Randomly initialize weights, between 0 and 1
        for i in range(0, self.NUM_BIASES):
            self.m_biases.append(random.uniform(0,1))

    def calculate_total_connections(self):#
        # Each neuron of the previous layer is connected to each neuron of the next layer
        return self.NUM_INPUTS * self.NUM_HIDDEN_NODES + \
                                   ((self.NUM_HIDDEN_LAYERS - 1) * (self.NUM_HIDDEN_NODES * self.NUM_HIDDEN_NODES)) + \
                                   self.NUM_HIDDEN_NODES * self.NUM_OUTPUTS

    def calculate_fitness(self, game_num):
        # Calculates the fitness in terms of the average chip counts ended up with after each game
        avg_chip_count = self.total_chip_count / game_num
        self.total_chip_count = 0
        return avg_chip_count

    def initialize_network(self):
        # Determine number of total connections
        self.TOTAL_CONNECTIONS = self.calculate_total_connections()

        # Initialise random weights and biases
        if self.training:
            self.init_weights()
            self.init_biases()

        self.reset_layers_to_zero()

    def reset_layers_to_zero(self):
        # Initialize the lists of 0's
        for i in range(1, self.NUM_HIDDEN_LAYERS + 1):
            self.m_layers[i] = list(np.zeros(self.NUM_HIDDEN_NODES))
        self.m_layers[self.NUM_HIDDEN_LAYERS + 1] = list(np.zeros(self.NUM_OUTPUTS))

    def run_network(self):
        # For layers from m_layers[1] to output layer, compute weighted sum of input from prev layer * weight + bias
        weight_boundary = 0
        bias_boundary = 0

        for layer in range(1, len(self.m_layers)):  # Iterate over the layers
            for inp in range(0, len(self.m_layers[layer - 1])):  # Inputs
                for neuron in range(0, len(self.m_layers[layer])):  # Neurons in curr layer
                    (self.m_layers[layer])[neuron] += (self.m_layers[layer - 1])[inp] * \
                                                       self.m_weights[weight_boundary + neuron]

                weight_boundary += len(self.m_layers[layer]) - 1

            # Add biases and do activations
            for neuron in range(0, len(self.m_layers[layer])):
                (self.m_layers[layer])[neuron] += self.m_biases[bias_boundary + neuron]

                # If we aren't at the output layer, apply the activation function
                if layer != len(self.m_layers) - 1:
                    (self.m_layers[layer])[neuron] = self.relu((self.m_layers[layer])[neuron])

            bias_boundary += len(self.m_layers[layer]) - 1

        # Output layer activations
        self.m_layers[len(self.m_layers) - 1] = self.softmax(self.m_layers[len(self.m_layers) - 1])

    def softmax(self, values):
        return np.exp(values) / sum(np.exp(values))

    def relu(self, val):
        if val > 0:
            return val
        else:
            return 0

    def compute_inputs(self):
        self.m_layers[0].clear()

        self.m_layers[0].append(self.chips_in_pot/300)
        self.m_layers[0].append(8/self.num_players_left)

        for chip_count in self.chip_counts:
            self.m_layers[0].append(chip_count / 100)

        self.m_layers[0].append(self.current_hand_strength)

        # If nothing in the aggression list, 0
        # If the player is folded, 0
        # Else add the average of their aggressions
        for a in self.opponent_aggressions:
            if len(self.opponent_aggressions[a]) == 0 \
                    or self.opponent_aggressions[a][len(self.opponent_aggressions[a])-1] == 0:
                self.m_layers[0].append(0)
            else:
                self.m_layers[0].append(np.average(self.opponent_aggressions[a]))

    def get_action(self):

        self.compute_inputs()
        self.reset_layers_to_zero()
        self.run_network()

        max_index = np.argmax(self.m_layers[len(self.m_layers)-1])

        if max_index == 0 or max_index == 1:  # Check or Call
            return 'call'
        elif max_index == 2 or max_index == 4:  # Raise or Bet
            return 'raise'
        elif max_index == 3: # Fold
            return 'fold'

        return 'error'

    def get_num_weights(self):
        return self.TOTAL_CONNECTIONS

    def get_num_biases(self):
        return self.NUM_BIASES

    def set_new_weight(self, add_value, weight_index):
        self.m_weights[weight_index] += add_value

    def set_new_bias(self, add_value, bias_index):
        self.m_biases[bias_index] += add_value

    def update_pot_count(self, pot_main_count):
        self.chips_in_pot = pot_main_count

    ###########################
    #                         #
    # BASEPOKERPLAYER METHODS #
    #                         #
    ###########################

    def declare_action(self, valid_actions, hole_card, round_state):
        # valid actions - 0 = raise info, 1 = call, 2 = fold

        # Update chip count
        self.update_chip_count(round_state["seats"])
        self.update_pot_count(round_state["pot"]["main"]["amount"])
        self.update_all_player_metrics(round_state["seats"])

        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        action = self.get_action()
        amount = 0

        if action == 'raise':
            amount = valid_actions[2]["amount"]["min"]

            if amount == -1:
                action = 'call'
                amount = (valid_actions[1]["amount"])

        elif action == 'call':
            amount = (valid_actions[1])["amount"]
        elif action == 'fold':
            amount = (valid_actions[0])["amount"]

        return action, amount  # action returned here is sent to the poker engine

    def update_chip_count(self, seats):
        for seat in seats:
            if seat["name"] == self.name:
                self.chip_count = seat["stack"]

    def update_total_chip_count(self):
        self.total_chip_count += self.chip_count

    def update_all_player_metrics(self, seats):
        count = 0
        self.chip_counts.clear()

        for seat in seats:
            # If the player is not folded, increment the tally
            if seat["stack"] > 0 or seat["state"] == "allin":
                count += 1

            # Update per player chip counts
            if seat["stack"] == 0:
                self.chip_counts.append(0)
            else:
                self.chip_counts.append(seat["stack"]/100)

        self.num_players_left = count

    def update_opponent_aggressions(self, action_info):
        self.aggression_vals = {'fold':0,
                                'check':0.3,
                                'call':0.65,
                                'raise':1}

        player = action_info['player_uuid']
        action = action_info['action']
        amount = action_info['amount']

        if self.opponent_aggressions.get(player) == None:
            return

        if action == 'call' and amount == '0':
            action = 'check'

        if len(self.opponent_aggressions[player]) == 10:
            self.opponent_aggressions[player].pop(0)

        self.opponent_aggressions[player].append(self.aggression_vals[action])

    def calculate_hole_strength(self, hole_cards):

        # Chen Formula - https://www.thepokerbank.com/strategy/basic/starting-hand-selection/chen-formula/
        score = 0

        # Score highest card
        best = 0
        for h in hole_cards:
            temp = self.card_values[h[1]]
            if temp > best:
                best = temp

        score += best

        # Multiply by two if pairs
        if hole_cards[0][1] == hole_cards[1][1]:
            score *= 2

        # Add 2 if the cards are same suit
        if hole_cards[0][0] == hole_cards[0][0]:
            score += 2

        # Subtract points according to the gap
        # 1 = -1, 2 = -2, 3 = -4, 4+ = -5
        h1_index = self.card_ranks.index(hole_cards[0][1])
        h2_index = self.card_ranks.index(hole_cards[1][1])
        diff = abs(h1_index - h2_index)

        if diff == 1:
            score -= 1
        elif diff == 2:
            score -= 2
        elif diff == 3:
            score -= 4
        elif diff >= 4:
            score -= 5

        # Straight bonus - if there's a 1 gap and both cards are lower than Q, +1
        if diff == 1:
            if (self.card_values[hole_cards[0][1]] and self.card_values[hole_cards[0][1]]) < self.card_values['Q']:
                score += 1

        # Round up and return - have to be careful we don't divide by 0
        if math.ceil(score) == 0:
            return 0
        else:
            return math.ceil(score) / 20

    def calculate_hand_strength(self, hole, community):
        if len(community) == 0:  # If we only have the hole cards available
            return self.calculate_hole_strength(hole)

        hand = list()
        board = list()

        for h in hole:
            hand.append(Card.new(h[1]+h[0].lower()))

        for c in community:
            board.append(Card.new(c[1] + c[0].lower()))

        e = self.hand_evaluator.evaluate(hand, board)

        return 1-(e / 7462)

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Update chip count at end of round to get most up to date, ONLY If final round

        if round_state["round_count"] == self.num_rounds:
            self.update_chip_count(round_state["seats"])
            self.update_total_chip_count()

    def receive_game_start_message(self, game_info):
        self.num_rounds = game_info["rule"]["max_round"]

        if not self.in_same_game_set:
            self.opponent_aggressions = {}

            # For X in seats, make dictionary entries
            for s in game_info["seats"]:
                # Make an entry w/ the key of the uuid, and the value as a list()
                if s["name"] != self.name:
                    self.opponent_aggressions[s["uuid"]] = list()

            self.in_same_game_set = True

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.current_hole = hole_card

    def receive_street_start_message(self, street, round_state):
        self.current_hand_strength = self.calculate_hand_strength(self.current_hole, round_state["community_card"])

    def receive_game_update_message(self, action, round_state):
        # If action was NOT done by this NN, update opponent aggression for the opponent who did the action
        self.update_opponent_aggressions(action)

