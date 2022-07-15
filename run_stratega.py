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
    #config = stratega.load_config('Stratega/resources/gameConfigurations/TBS/Tests/OpenTheDoor.yaml')

    log_path = './sgaLog.yaml'
    stratega.set_default_logger(log_path)

    number_of_games = 1
    player_count = 2
    seed = 42

    # resolution = stratega.Vector2i(1920, 1080)
    # runner = stratega.create_runner(config)

    # # #runner.play([MCGSAgent(seed=seed), RandomAgent(seed=seed)], resolution, seed)
    # runner.play([MCTSAgent(seed=seed), "MCTSAgent"], resolution, seed)


    arena = stratega.create_arena(config)

    #arena.run_games(player_count, seed, number_of_games, 1, [MCGSAgent(seed=seed), "MCTSAgent"])
    arena.run_games(player_count, seed, number_of_games, 1, [MCTSAgent(seed=seed), "MCTSAgent"])

