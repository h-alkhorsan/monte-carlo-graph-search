import stratega 
import numpy as np

class NoveltyHeuristic:
    def __init__(self, gs, graph):
        self.gs = gs
        self.graph = graph 

    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0
        observation = gs.print_board()
        if not self.graph.has_node(observation):
            score += 10
        return score 



class MinimizeDistanceHeuristic(stratega.MinimizeDistanceHeuristic):
    def __init__(self):
        stratega.MinimizeDistanceHeuristic.__init__(self)


class RelativeStrengthHeuristic:
    def __init__(self, gs):
        self.gs = gs 

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

        player_entity_count = 0
        opponent_entity_count = 0

        observation = gs.print_board()
        observation = np.array(observation.split()).reshape(gs.get_board_width(), gs.get_board_height())
        for i in range(gs.get_board_width()):
            for j in range(gs.get_board_height()):
                if observation[i][j] in player_entities:
                    #player_entity_count += 1
                    score += 10
                if observation[i][j] in opponent_entities:
                    score -= 10
                    #opponent_entity_count += 1
        
        return score 


class OpenTheDoorHeuristic:
    def __init__(self, gs):
        self.gs = gs
    
    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0
        if gs.is_game_over():
            if gs.get_winner_id() == player_id:
                score += 100
            else:
                score -= 100

        observation = gs.print_board()
        observation = np.array(observation.split()).reshape(gs.get_board_width(), gs.get_board_height())
        key_count = 0
        for i in range(gs.get_board_width()):
            for j in range(gs.get_board_height()):
                if observation[i][j] == 'k':
                    key_count += 1
        return score 



class GeneralHeuristic:
    def __init__(self, gs):
        self.gs = gs
    
    def evaluate_gamestate(self, forward_model, gs, player_id):
        score = 0.0
        if gs.is_game_over():
            if gs.get_winner_id() == player_id:
                score += 100
            else:
                score -= 100

        return score 
