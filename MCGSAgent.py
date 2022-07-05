import stratega
import numpy as np
import math
from copy import deepcopy
from MCGSGraph import Graph


class MinimizeDistanceHeuristic(stratega.MinimizeDistanceHeuristic):
    def __init__(self):
        stratega.MinimizeDistanceHeuristic.__init__(self)

class Timer(stratega.Timer):
    def __init__(self):
        stratega.Timer.__init__(self)

class MCGSAgent(stratega.Agent):

    def __init__(self, seed, budget_type="MAX_FM_CALLS"):
        stratega.Agent.__init__(self, "MCGSAgent")
        self.random = np.random.RandomState(seed)
        self.graph = Graph(seed)
        self.budget_type = budget_type

    # def heuristic(self, gs):

    #     score = 0.0
    #     if gs.is_game_over():
    #         if gs.get_winner_id() == self.get_player_id():
    #             score += 2.0
    #         else:
    #             score -= 2.0
    #         return score 

    #     king_x, king_y = -1, -1
    #     total_distance, mean_distance = 0.0, 0.0

    #     positions = {}
    #     opponent_entities = []
    #     player_entities = []

    #     king_hp = 200.0

    #     for entity in gs.get_entities():
    #         positions[entity.get_id()] = entity.get_position()
    #         if entity.get_owner_id() != gs.get_current_tbs_player():
    #             opponent_entities.append(entity.get_id())
    #             entity_type = gs.get_game_info().get_entity_type(entity.get_entity_type_id())
    #             if entity_type.get_name() == "King":
    #                 king_x, king_y = entity.x(), entity.y()
    #                 opponent_king_hp = entity.get_parameter("Health")
    #         else:
    #             player_entities.append(entity.get_id())

    #     for entity in player_entities:
    #         total_distance += abs(positions[entity].x - king_x) + abs(positions[entity].y - king_y)

    #     mean_distance = total_distance / len(player_entities)

    #     maximum_distance = 400.0
    #     score = 1.0 - mean_distance / maximum_distance
    #     score += 1.0 - opponent_king_hp / 400.0

    #     total_n_entities = 20.0
    #     score -= len(player_entities) / total_n_entities
    #     score -= len(opponent_entities) / total_n_entities
    #     return (score+2.0) / 4.0


    def init(self, gs, forward_model, timer):
        self.node_counter = 0
        self.edge_counter = 0
        self.num_rollouts = 8
        self.rollout_depth = 50

        self.root_node = Node(id=self.get_observation(gs), parent=None, is_leaf=True, action=None, reward=0, visits=0, done=False)
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
        self.set_root_node(gs)
        self.graph.reroute_all()

        while not self.is_budget_over():

            selection_env = deepcopy(gs)
            node = self.selection(selection_env, forward_model)
         
            if node is None:
                children, actions_to_children = [self.root_node], [None] 

            else:
                children, actions_to_children = self.expansion(node, selection_env, forward_model)
                
          
            for idx in range(len(children)):

                child_average_reward, novelties = self.sequential_simulation(actions_to_children[idx], selection_env, forward_model)
                self.num_simulations += 1      
                # rollout_nodes, rewards = self.add_novelties_to_graph(novelties)
                # for i, node in enumerate(rollout_nodes):
                #     self.back_propagation(node, rewards[i])

                self.back_propagation(children[idx], child_average_reward)
                
            self.num_iterations += 1

        best_node, action = self.select_best_step(self.root_node, closest=False)
       
        return action

    def selection(self, env, forward_model):

        if self.root_node.is_leaf:
            return self.root_node

        node = self.graph.select_frontier_node(noisy=True)
        if node is None:
            return self.root_node

        selected_node = self.go_to_node(node, env, forward_model)
        return selected_node

    def go_to_node(self, destination_node, env, forward_model): 

        # TODO: For stochastic, continuously go to the node

        observation = self.get_observation(env)
        node = self.graph.get_node_info(observation)

        reached_destination = False

        while self.graph.has_path(node, destination_node) and not reached_destination:
            
            observations, actions = self.graph.get_path(node, destination_node)
            for action in actions:
                if action.validate(env) == False:
                    continue

            for idx, action in enumerate(actions):

                previous_observation = self.get_observation(env)
                parent_node = self.graph.get_node_info(previous_observation)
                
                forward_model.advance_gamestate(env, action)
                self.forward_model_calls += 1
                reward = self.evaluate_state(forward_model, env, self.get_player_id())
                done = self.is_game_over(env)

                current_observation = self.get_observation(env)
                
                if not self.graph.has_node(current_observation):
                    self.add_new_observation(current_observation, parent_node, action, reward, done)

                elif not self.graph.has_edge_by_nodes(parent_node, self.graph.get_node_info(current_observation)):
                    self.add_edge(parent_node, self.graph.get_node_info(current_observation), action, reward, done)

                if observations[idx + 1] != current_observation:
                    node = self.graph.get_node_info(current_observation)
                    break
              
                if destination_node.id == self.get_observation(env):
                    reached_destination = True
                    break
       
        return self.graph.get_node_info(self.get_observation(env))

    def expansion(self, node, env, forward_model):

     
        new_nodes = []
        actions_to_new_nodes = []

        # Nodes might not be leaves due to action_failure
        if node.is_leaf:
            node.is_leaf = False
            self.graph.remove_from_frontier(node)

        actions = forward_model.generate_actions(env, self.get_player_id())

        for action in actions:
            if action.validate(env) == False:
                assert False, "failed action in expansion"


            expansion_env = deepcopy(env)

            forward_model.advance_gamestate(expansion_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, expansion_env, self.get_player_id())
            done = self.is_game_over(expansion_env)
          
            current_observation = self.get_observation(expansion_env)
            child, reward = self.add_new_observation(current_observation, node, action, reward, done)
        

            if child is not None:
  
                new_nodes.append(child)
                actions_to_new_nodes.append(action)

        return new_nodes, actions_to_new_nodes


    def sequential_simulation(self, action_to_node, env, forward_model):
        rewards = []
        paths = []

        for i in range(self.num_rollouts):
            possible_actions = forward_model.generate_actions(env, self.get_player_id())
            action_list = self.random.choice(possible_actions, self.rollout_depth)
            average_reward, path = self.rollout(action_to_node, env, forward_model, action_list)

            paths.append(path)
            rewards.append(average_reward)

        return np.mean(rewards), paths

   
    def rollout(self, action_to_node, env, forward_model, action_list):
    
        cum_reward = 0
        path = []
        rollout_env = deepcopy(env)
        forward_model.advance_gamestate(rollout_env, action_to_node)
        self.forward_model_calls += 1

        previous_observation = self.get_observation(rollout_env)

        for idx, action in enumerate(action_list):
            if action.validate(rollout_env) == False:
                #assert False, "failed action in rollout"
                continue
 
            forward_model.advance_gamestate(rollout_env, action)
            self.forward_model_calls += 1
            reward = self.evaluate_state(forward_model, rollout_env, self.get_player_id())
            done = self.is_game_over(rollout_env)

            observation = self.get_observation(rollout_env)
    
            cum_reward += reward
        
            path.append((previous_observation, observation, action, reward, done))
            previous_observation = observation

        return cum_reward, path

    def back_propagation(self, node, reward):
    
        while node is not None:
            node.visits += 1
            node.total_value += reward 
            node = node.parent

    def add_new_observation(self, current_observation, parent_node, action, reward, done):

        new_node = None

        if current_observation != parent_node.id:  # don't add node if nothing has changed in the observation
            if self.graph.has_node(current_observation) is False:  # if the node is new, create it and add to the graph
                child = Node(id=current_observation, parent=parent_node,
                             is_leaf=True, action=action, reward=reward, visits=0, done=done)
                self.add_node(child)
                new_node = child
            else:
                child = self.graph.get_node_info(current_observation)

                if child.is_leaf: #enable for FMC optimisation, comment for full exploration
                    new_node = child

            edge = self.add_edge(parent_node, child, action, reward, done)
   
        return new_node, reward


    def get_optimal_action(self, node):

        new_root_node, action = self.select_best_step(node)
        new_root_node.chosen = True
        new_root_node.parent = None

        if self.graph.has_path(new_root_node, self.root_node):
            self.graph.reroute_path(new_root_node, self.root_node)
            self.root_node.action = self.graph.get_edge_info(self.root_node.parent, self.root_node).action

        self.root_node = new_root_node
    
        return action

    def set_root_node(self, gs):

        old_root_node = self.root_node

        new_root_id = self.get_observation(gs)

        if not self.graph.has_node(new_root_id):
            self.root_node = Node(id = self.get_observation(gs), parent=None, is_leaf=True, action=None, reward=0, visits=0, done=False)
            self.add_node(self.root_node)

        else:
            self.root_node = self.graph.get_node_info(new_root_id)

        self.graph.set_root_node(self.root_node)

        if self.root_node.id != old_root_node.id:
            self.root_node.chosen = True
            self.root_node.parent = None

            # Reroute the old root node
            if self.graph.has_path(self.root_node, old_root_node):
                self.graph.reroute_path(self.root_node, old_root_node)
                old_root_node.action = self.graph.get_edge_info(old_root_node.parent, old_root_node).action

    

    def select_best_step(self, node, closest=False):

        best_node = None
        if closest:
            best_node = self.graph.get_closest_done_node(only_reachable=True)

        if best_node is None:
            best_node = self.graph.get_best_node(only_reachable=True)

                
        if best_node is None:
            return self.root_node, self.root_node.action
            


        while best_node.parent != self.root_node:
            best_node = best_node.parent

        edge = self.graph.get_edge_info(node, best_node)  # pick the edge between children

        return best_node, edge.action

    def check_paths(self):
        self.graph.reroute_paths(self.root_node)

    def add_node(self, node):

        if not self.graph.has_node(node):
            self.graph.add_node(node)
            self.graph.add_to_frontier(node)
            self.node_counter += 1


    def add_edge(self, parent_node, child_node, action, reward, done):

        edge = Edge(id=self.edge_counter, node_from=parent_node, node_to=child_node,
                    action=action, reward=reward, done=done)

        if not self.graph.has_edge(edge):
            self.graph.add_edge(edge)
            self.edge_counter += 1

           
        if child_node.unreachable is True and child_node != self.root_node:  # if child was unreachable make it reachable through this parent
            child_node.set_parent(parent_node)
            child_node.action = action
            child_node.unreachable = False

        return edge


    def add_novelties_to_graph(self, paths):

        nodes = []
        node_rewards = []
        for path in paths:
            for idx, step in enumerate(path):

                observation = step[1]
                if self.graph.has_node(observation) is False:

                    for i in range(idx + 1):
                        step_i = path[i]
                        previous_observation = step_i[0]
                        current_observation = step_i[1]
                        action = step_i[2]
                        reward = step_i[3]
                        done = step_i[4]
                
                        
                        parent_node = self.graph.get_node_info(previous_observation)
                        if parent_node.unreachable and parent_node != self.root_node:
                            print("No way novelty!")
                            assert False
                        node, node_reward = self.add_new_observation(current_observation, parent_node, action, reward, done)
                        nodes.append(node)
                        node_rewards.append(node_reward)

                 
        return nodes, node_rewards

class Node:

    def __init__(self, id, parent, is_leaf, action, reward, visits, done):

        self.id = id
        self.parent = None
        self.set_parent(parent)

        self.action = action
        self.total_value = reward
        self.visits = visits
        self.is_leaf = is_leaf
        #self.novelty_value = novelty_value
        self.done = done
        self.chosen = False
        self.unreachable = False

    def uct_value(self):
        c = 1 / math.sqrt(2)
        ucb = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return self.value() + ucb

    def value(self):
        if self.visits == 0:
            return 0
        else:
            return self.total_value / self.visits

    def trajectory_from_root(self):

        action_trajectory = []
        current_node = self

        while current_node.parent is not None:
            action_trajectory.insert(0, current_node.action)
            current_node = current_node.parent

        return action_trajectory

    def reroute(self, path, actions):
        parent_order = list(reversed(path))
        actions_order = list(reversed(actions))
        node = self

        for i in range(len(parent_order) - 1):

            if node.parent != parent_order[i + 1]:
                node.set_parent(parent_order[i + 1])
            node.action = actions_order[i]
            node = parent_order[i + 1]

    def set_parent(self, parent):
        self.parent = parent

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
    runner.play([MCGSAgent(seed=seed), "MCTSAgent"], resolution, seed)

    #arena = stratega.create_arena(config)
    #arena.run_games(player_count, seed, number_of_games, 1, [MCGSAgent(seed=seed), "MCTSAgent"])

