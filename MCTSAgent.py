import stratega
import numpy as np
import math
import csv
from copy import deepcopy
from MCTSGraph import Graph
from utils import Timer 
from stratega_heuristics import *
from opponent_models import *

class MCTSAgent(stratega.Agent):

    def __init__(self, seed, budget_type=None, game_type=None, opponent_name=None):
        stratega.Agent.__init__(self, "MCTSAgent")
        self.seed = seed 
        self.budget_type = budget_type
        self.game_type = game_type
        self.opponent_name = opponent_name

    def init(self, gs, forward_model, timer):
        print("init MCTSAgent")
        self.random = np.random.RandomState(self.seed)
        self.graph = Graph()
        self.node_counter = 0
        self.edge_counter = 0
        self.num_rollouts = 8
        self.invalid_action_count = 0
        self.n_turns = 0

        self.use_opponent_model = True 
        observation = self.get_observation(gs)
        self.root_node = Node(id=self.node_counter, observation=observation, parent=None, is_leaf=True, value=0, visits=0, redundant=False, is_root=True)
        self.add_node(self.root_node)
        self.root_node.chosen = True 
 
        self.num_simulations = 0
        self.forward_model_calls = 0        
        self.max_forward_model_calls = 1000

        self.num_iterations = 0
        self.max_iterations = 100

        self.timer = Timer()
        self.max_time_ms = 10000

        self.heuristic = GeneralHeuristic(gs)


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

    def get_opponent_id(self):
        if self.get_player_id() == 0:
            return 1
        else:
            return 0

    def get_similar_action(self, failed_action, possible_actions):
        similar_action = None
        for action in possible_actions:
            if action.get_action_name() == failed_action.get_action_name():
                similar_action = action 

        if similar_action == None:
            similar_action = possible_actions[-1]

        return similar_action

    def log(self):
        data = [
            self.seed,
            self.game_type, 
            self.forward_model_calls,
            self.num_iterations,
            self.node_counter,
            self.invalid_action_count,
            self.n_turns,
            self.opponent_name,
            int(self.use_opponent_model),
            self.heuristic.__str__(),
        ]

        with open(f'experiments/seed_{self.seed}_MCTS.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def compute_action(self, gs, forward_model, timer, draw_graph=True, log_data=False):
        print("MCTS Action")
        self.n_turns += 1
        self.reset_budget()
        possible_actions = forward_model.generate_actions(gs, self.get_player_id())

        if len(possible_actions) == 1:
            return stratega.ActionAssignment.from_single_action(possible_actions[0])

        action = self.plan(gs, forward_model)
            
        if action.validate(gs) == False:
            self.invalid_action_count += 1
            print("MCTS: invalid action...")
            action = self.get_similar_action(action, possible_actions)

        action_assignment = stratega.ActionAssignment.from_single_action(action)
                
        if draw_graph:
            self.graph.draw_graph()

        if log_data:
            self.log()

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

        action = self.get_best_action(self.root_node)
       
        return action


    def selection(self, node, env, forward_model):
        while not node.is_leaf:
            node, action = self.select_child(node)
            forward_model.advance_gamestate(env, action)
            self.forward_model_calls += 1
        return node 

    def expansion(self, node, env, forward_model):
        children = []
        node.is_leaf = False 

        actions = forward_model.generate_actions(env, self.get_player_id())
        for action in actions:

            expansion_env = deepcopy(env)
            forward_model.advance_gamestate(expansion_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, env, self.get_player_id())
           
            observation = self.get_observation(expansion_env)
        
            child = Node(id=self.node_counter, observation=observation, parent=node, is_leaf=True, value=0, visits=0, redundant=self.graph.has_observation(observation), is_root=False)
            edge = Edge(id=self.edge_counter, node_from=node, node_to=child, action=action, reward=reward)

            self.node_counter += 1
            self.edge_counter += 1

            self.add_node(child)
            children.append(child)

            self.graph.add_edge(edge)

        return children 


    def simulation(self, node, env, forward_model):
        rewards = []
        action = self.graph.get_edge_info(node.parent, node).action
        simulation_env = deepcopy(env)
        forward_model.advance_gamestate(simulation_env, action)

        for i in range(self.num_rollouts):
            average_reward = self.rollout(simulation_env, forward_model)
            rewards.append(average_reward)

        return np.mean(rewards)

    def rollout(self, env, forward_model):
        cum_reward = 0
        rollout_env = deepcopy(env)

        while True:
            actions = forward_model.generate_actions(rollout_env, self.get_player_id())
            
            if len(actions) == 0:
                break 

            if self.is_budget_over():
                break 

            if self.is_game_over(rollout_env):
                break 

            random_action = self.random.choice(actions)
            forward_model.advance_gamestate(rollout_env, random_action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, rollout_env, self.get_player_id())
            cum_reward += reward 

            # random opponent model
            if self.use_opponent_model:
                rollout_env.set_current_tbs_player(self.get_opponent_id())

                opponent_actions = forward_model.generate_actions(rollout_env, self.get_opponent_id())
                random_opponent_action = self.random.choice(opponent_actions)
                forward_model.advance_gamestate(rollout_env, random_opponent_action)
                self.forward_model_calls += 1

                rollout_env.set_current_tbs_player(self.get_player_id())

        return cum_reward


    def back_propagation(self, node, value):
        while True:
            node.visits += 1
            node.value += value 
            if node.chosen:
                break 
            node = node.parent

    def get_best_action(self, node):
        self.root_node.is_root = False 
        
        new_root_node, action = self.select_child(node)
        new_root_node.chosen = True 
        new_root_node.is_root = True
         
        self.root_node = new_root_node
        return action

    def add_node(self, node):
        self.graph.add_node(node)

    def select_child(self, node):
        children = self.graph.get_children(node)

        child_values = [child.uct_value() for child in children]
    
        child = children[child_values.index(max(child_values))]
        edge = self.graph.get_edge_info(node, child)

        return child, edge.action

    

class Node:

    def __init__(self, id, observation, parent, is_leaf, value, visits, redundant, is_root):

        self.id = id
        self.observation = observation
        self.parent = parent
        self.value = value
        self.visits = visits
        self.is_leaf = is_leaf
        self.redundant = redundant
        self.is_root = is_root

        self.chosen = False
        self.unreachable = False
        

    def uct_value(self):
        ucb_const = 1 / math.sqrt(2)
        mean = self.value / self.visits
        ucb = ucb_const * math.sqrt(math.log(self.parent.visits if self.parent is not None else 1 + 1) / self.visits)
        return mean + ucb

    def __hash__(self):
        return hash(self.id)


class Edge:

    def __init__(self, id, node_from, node_to, action, reward):
        self.id = id
        self.node_from = node_from
        self.node_to = node_to
        self.action = action
        self.reward = reward

    def __hash__(self):
        return hash(self.id)


