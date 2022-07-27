import stratega
import numpy as np
from MCTSAgent import MCTSAgent
from MCGSAgent import MCGSAgent

### Random agent for testing ###
class RandomAgent(stratega.Agent):
    def __init__(self, seed):
        stratega.Agent.__init__(self, "RandomAgent")
        self.random = np.random.RandomState(seed)

    def init(self, gs, forward_model, timer):
        print("init RandomAgent")

    def compute_action(self, gs, forward_model, timer):
        actions = forward_model.generate_actions(gs, self.get_player_id())
        random_action = self.random.choice(actions)

        action_assignment = stratega.ActionAssignment.from_single_action(random_action)
        return action_assignment

if __name__ == '__main__':

    config = stratega.load_config('Stratega/resources/gameConfigurations/TBS/Tests/OpenTheDoor.yaml')
    seed=42
    
    log_path = f'experiments/SGALOG_{seed}_MCGSvsMCTS_OTD.yaml'
    maps_path = ""
    stratega.set_default_logger(log_path)

    number_of_games = 5
    player_count = 2

    budget_type = "MAX_TIME_MS"
    game_type = "OTD"

    arena = stratega.create_arena(config)

    if not maps_path:
        arena.run_games(player_count, seed, number_of_games, 1, [
            MCGSAgent(seed=seed, budget_type=budget_type, game_type=game_type, opponent_name="MCTS"), 
            MCTSAgent(seed=seed, budget_type=budget_type, game_type=game_type, opponent_name="MCGS")])
    else:
        config.level_definitions = stratega.load_levels_from_yaml(maps_path, config)
        map_number = len(config.level_definitions)
        arena.run_games(player_count, seed, number_of_games, 1, [
            MCGSAgent(seed=seed, budget_type=budget_type, game_type=game_type, opponent_name="MCTS"), 
            MCTSAgent(seed=seed, budget_type=budget_type, game_type=game_type, opponent_name="MCGS")])

