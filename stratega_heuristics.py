import stratega 
import numpy as np
import math

class MinimizeDistanceHeuristic(stratega.MinimizeDistanceHeuristic):
    def __init__(self):
        stratega.MinimizeDistanceHeuristic.__init__(self)
    def __str__(self):
        return "Minimize Distance Heuristic"

class RelativeStrengthHeuristic:
    def __init__(self, gs):
        self.gs = gs 
    
    def __str__(self):
        return "Relative Strength Heuristic"


    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0 
        if gs.is_game_over():
            if gs.get_winner_id() == player_id:
                score += 100
            else:
                score -= 100


        if player_id == 0:
            player_entities = ['a0', 'h0', 'w0']
            opponent_entities = ['a1', 'h1', 'w1']

        else:
            player_entities = ['a1', 'h1', 'w1']
            opponent_entities = ['a0', 'h0', 'w0']

        observation = gs.print_board()
        observation = np.array(observation.split()).reshape(gs.get_board_width(), gs.get_board_height())
        for i in range(gs.get_board_width()):
            for j in range(gs.get_board_height()):
                if observation[i][j] in player_entities:
            
                    score += 10
                if observation[i][j] in opponent_entities:
                    score -= 10
        
        return score 


class OpenTheDoorHeuristic:
    def __init__(self, gs):
        self.gs = gs
    
    def __str__(self):
        return "Open The Door Heuristic"

    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0
        if gs.is_game_over():
            if gs.get_winner_id() == player_id:
                score += 100
            else:
                score -= 100

        observation = gs.print_board()
        observation = np.array(observation.split()).reshape(gs.get_board_width(), gs.get_board_height())

        key_location = np.argwhere(observation == 'k')
        player_location = np.argwhere(observation == 's0')

        if key_location.any():
            euclidean_dist = np.linalg.norm(key_location - player_location)
            score -= euclidean_dist

        else:
            score += 25

        return score 

class GeneralHeuristic:
    def __init__(self, gs):
        self.gs = gs
    
    def __str__(self):
        return "General Heuristic"
    
    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0
        if gs.is_game_over():
            if gs.get_winner_id() == player_id:
                score += 100
            else:
                score -= 100

        return score 
