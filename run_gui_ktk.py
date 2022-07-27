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
    
    config = stratega.load_config('Stratega/resources/gameConfigurations/TBS/Original/KillTheKing.yaml')
    seed=42

    runner = stratega.create_runner(config)
    resolution = stratega.Vector2i(1920, 1080)    

    budget_type = "MAX_TIME_MS"
    game_type = "KTK"

    runner.play([
        MCGSAgent(seed=seed, budget_type=budget_type, game_type=game_type), 
        MCTSAgent(seed=seed, budget_type=budget_type, game_type=game_type)], resolution, seed)

    # print winner
    gs = runner.get_gamestate()
    print(gs.get_winner_id())


