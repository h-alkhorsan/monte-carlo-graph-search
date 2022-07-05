import stratega
import numpy as np
import math
from copy import deepcopy
from MCTSGraph import Graph


class MinimizeDistanceHeuristic(stratega.MinimizeDistanceHeuristic):
    def __init__(self):
        stratega.MinimizeDistanceHeuristic.__init__(self)

class Timer(stratega.Timer):
    def __init__(self):
        stratega.Timer.__init__(self)

class MCTSAgent(stratega.Agent):

    def __init__(self, seed, budget_type="MAX_FM_CALLS"):
        stratega.Agent.__init__(self, "MCTSAgent")
        self.random = np.random.RandomState(seed)
        self.graph = Graph()
        self.budget_type = budget_type

    def init(self, gs, forward_model, timer):
        self.node_counter = 0
        self.edge_counter = 0
        self.num_rollouts = 8
        self.rollout_depth = 50

        self.root_node = Node(id=self.get_observation(gs), parent=None, is_leaf=True, value=0, visits=0, done=False)
        self.add_node(self.root_node)
        self.root_node.chosen = True 
 
        self.num_simulations = 0
        self.forward_model_calls = 0        
        self.max_forward_model_calls = 1000

        self.num_iterations = 0
        self.max_iterations = 100

        self.timer = Timer()
        self.max_time_ms = 1000

        self.heuristic = MinimizeDistanceHeuristic()


    def is_budget_over(self):
        if self.budget_type == "MAX_FM_CALLS":
            return self.forward_model_calls >= self.max_forward_model_calls
        elif self.budget_type == "MAX_ITERATIONS":
            return self.num_iterations >= self.max_iterations
        elif self.budget_type == "MAX_TIME_MS":
            return self.timer.elapsed_milliseconds() >= self.max_time_ms
        else:
            return False 

    def reset_budget(self):
        self.forward_model_calls = 0
        self.num_iterations = 0
        self.timer = Timer()


    def evaluate_state(self, forward_model, gs, player_id):
        return self.heuristic.evaluate_gamestate(forward_model, gs, player_id)    

    def is_game_over(self, gs):
        return gs.is_game_over()

    def get_observation(self, gs):
        return gs.print_board()

    def compute_action(self, gs, forward_model, timer, draw_graph=False):
        self.reset_budget()
        
        action = self.plan(gs, forward_model)
            
        if action.validate(gs) == False:
                print("invalid action... ending turn")
                end_action_assignment = stratega.ActionAssignment.create_end_action_assignment(self.get_player_id())
                return end_action_assignment

        action_assignment = stratega.ActionAssignment.from_single_action(action)
                
        if draw_graph:
            self.graph.draw_graph()

        return action_assignment


    
    def plan(self, gs, forward_model): 
        while not self.is_budget_over():

            selection_env = deepcopy(gs)
            node = self.selection(self.root_node, selection_env, forward_model)
         
            children = self.expansion(node, selection_env, forward_model)

            for child in children:
                value = self.simulation(child, selection_env, forward_model)
                self.back_propagation(child, value)

                
            self.num_iterations += 1

        action = self.get_optimal_action(self.root_node)
       
        return action


    def selection(self, node, env, forward_model):
        while not node.is_leaf:
            node, action = self.select_child(node)
            forward_model.advance_famestate(env, action)
        return node 

    def expansion(self, node, env, forward_model):
        children = []
        node.is_leaf = False 

        actions = forward_model.generate_actions(env, self.get_player_id())
        for action in actions:
            if action.validate(env) == False:
                print("invalid action in expansion")
                continue
            expansion_env = deepcopy(env)
            forward_model.advance_gamestate(expansion_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, env, self.get_player_id())
            done = self.is_game_over(expansion_env)
            observation = self.get_observation(expansion_env)

            child = Node(id=observation, parent=node, is_leaf=True, value=0, visits=0, done=done)
            edge = Edge(id=self.edge_counter, node_from=node, node_to=child, action=action, reward=reward, done=done)

            self.node_counter += 1
            self.edge_counter += 1

            self.add_node(child)
            children.append(child)

            self.graph.add_edge(edge)

        return children 


    def simulation(self, node, env, forward_model):
        rewards = []
        action = self.graph.get_edge_info(node.parent, node).action
        for i in range(self.num_rollouts):
            actions = forward_model.generate_actions(env, self.get_player_id())
            action_list = self.random.choice(actions, self.rollout_depth)
            average_reward = self.rollout(action, env, action_list, forward_model)
            rewards.append(average_reward)
        return np.mean(rewards)

    def rollout(self, action, env, action_list, forward_model):
        cum_reward = 0
        rollout_env = deepcopy(env)
        forward_model.advance_gamestate(rollout_env, action)
        self.forward_model_calls += 1

        for action in action_list:
            if action.validate(rollout_env) == False:
                #print("invalid action in rollout")
                continue
            forward_model.advance_gamestate(rollout_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, rollout_env, self.get_player_id()) 
            done = self.is_game_over(rollout_env)

            cum_reward += reward
            if done:
                break 

        return cum_reward


    def back_propagation(self, node, value):
        while True:
            node.visits += 1
            node.value += value 
            if node.chosen:
                break 
            node = node.parent

    def get_optimal_action(self, node):
        new_root_node, action = self.select_child(node)
        new_root_node.chosen = True 
        self.root_node = new_root_node
        return action

    def add_node(self, node):
        self.graph.add_node(node)

    def select_child(self, node):
        children = self.graph.get_children(node)
    
        edges = []
        for child in children:
            edges.append(self.graph.get_edge_info(node, child))

        child_values = [child.uct_value() for child in children]
    
        child = children[child_values.index(max(child_values))]
        edge = self.graph.get_edge_info(node, child)

        return child, edge.action

    

class Node:

    def __init__(self, id, parent, is_leaf, value, visits, done):

        self.id = id
        self.parent = parent
        self.value = value
        self.visits = visits
        self.is_leaf = is_leaf
        #self.novelty_value = novelty_value
        self.done = done

        self.chosen = False
        self.unreachable = False

    def uct_value(self):
        c = 1 / math.sqrt(2)
        mean = self.value / self.visits
        ucb = c * math.sqrt(math.log(self.parent.visits if self.parent is not None else 1 + 1) / self.visits)
        return mean + ucb

    def __hash__(self):
        return hash(self.id)


class Edge:

    def __init__(self, id, node_from, node_to, action, reward, done):
        self.id = id
        self.node_from = node_from
        self.node_to = node_to
        self.action = action
        self.reward = reward
        self.done = done

    def __hash__(self):
        return hash(self.id)



if __name__ == '__main__':
    
    config = stratega.load_config('Stratega/resources/gameConfigurations/TBS/Original/KillTheKing.yaml')
    #config = stratega.load_config('Stratega/resources/gameConfigurations/TBS/Tests/OpenTheDoor.yaml')
 
    log_path = './sgaLog.yaml'
    stratega.set_default_logger(log_path)
    
    number_of_games = 5
    player_count = 2
    seed = 42

    resolution = stratega.Vector2i(1920, 1080)
    runner = stratega.create_runner(config)
    runner.play([MCTSAgent(seed=seed), "MCTSAgent"], resolution, seed)

   # arena = stratega.create_arena(config)
   # arena.run_games(player_count, seed, number_of_games, 1, [MCTSAgent(seed=seed), "MCTSAgent"])