import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import pygraphviz_layout
from matplotlib.lines import Line2D

class Graph:

    def __init__(self, seed):
        self.graph = nx.DiGraph()
        self.frontier = []

        self.amplitude_factor = 0.2
        self.noisy_min_value = 0.0001

        self.root_node = None
        self.new_nodes = []
        self.random = np.random.RandomState(seed)

    def add_node(self, node):
        self.graph.add_node(node.id, info=node)

    def add_edge(self, edge):
        self.graph.add_edge(edge.node_from.id, edge.node_to.id, info=edge)

    def add_to_frontier(self, node):
        self.frontier.append(node)
        self.new_nodes.append(node)

    def remove_from_frontier(self, node):
        self.frontier.remove(node)

    def in_frontier(self, node):
        return node in self.frontier

    def select_frontier_node(self):

        selectable_nodes = [x for x in self.frontier if x.unreachable is False]
        if len(selectable_nodes) == 0:
            print("no selectable nodes")
            return None
 
        amplitude = self.get_best_node().uct_value() * self.amplitude_factor
        noise = self.random.normal(0, max(amplitude, self.noisy_min_value), len(selectable_nodes))
     
        best_node = selectable_nodes[0]
        best_node_value = best_node.uct_value() + noise[0] 

        for i, n in enumerate(selectable_nodes):
            if n.uct_value()  > best_node_value:
                best_node = n
                best_node_value = n.uct_value() + noise[i] 
       
        assert self.has_path(self.root_node, best_node), "no path to best node"

        return best_node

    def set_root_node(self, root_node):
        self.root_node = root_node

    def reroute_paths(self, root_node):
        for node_id, node in self.graph.nodes.data('info'):
            if root_node.id != node_id:
                if self.has_path(self.root_node, node):
                    self.reroute_path(self.root_node, node)
                    node.unreachable = False
                else:
                    node.unreachable = True

    def reroute_path(self, node_from, node_to):
        nodes, actions = self.get_path(node_from, node_to)
        node_path = [self.get_node_info(x) for x in nodes]
        node_to.reroute(node_path, actions)

    def get_path(self, node_from, node_to):
        observations = nx.dijkstra_path(self.graph, node_from.id, node_to.id)
        actions = []
        for i in range(len(observations) - 1):
            actions.append(self.get_edge_info(self.get_node_info(observations[i]), self.get_node_info(observations[i + 1])).action)

        return observations, actions

    def has_path(self, node_from, node_to):
        return nx.has_path(self.graph, node_from.id, node_to.id)

    def get_node_info(self, id):
        return self.graph.nodes[id]['info']

    def get_all_nodes_info(self):
        return list(nx.get_node_attributes(self.graph, 'info').values())

    def get_best_node(self):

        nodes = self.get_all_nodes_info()
        nodes.remove(self.root_node)

        selectable_nodes = [x for x in nodes if x.unreachable is False]
   
        if len(selectable_nodes) > 0:
            best_node = selectable_nodes[0]
            best_node_value = best_node.value() + self.get_edge_info(best_node.parent, best_node).reward
        else:
            best_node = None
            best_node_value = None

        for n in selectable_nodes:
            selected_node_value = n.value() + self.get_edge_info(n.parent, n).reward
            if best_node_value < selected_node_value:
                best_node = n
                best_node_value = selected_node_value

        return best_node

    def has_node(self, id):
        return self.graph.has_node(id)

    def has_edge(self, edge):
        parent = edge.node_from
        child = edge.node_to
        return self.graph.has_edge(parent.id, child.id)

    def has_edge_by_nodes(self, node_from, node_to):
        return self.graph.has_edge(node_from, node_to)

    def get_children(self, node):
        children = []
        for n in self.graph.successors(node.id):
            child_node = self.graph.nodes[n]['info']
            children.append(child_node)
        return children

    def get_edge_info(self, parent, child):
        return self.graph.get_edge_data(parent.id, child.id)['info']

    def reroute_all(self):
      
        all_nodes = self.get_all_nodes_info()
        for n in all_nodes:
            n.unreachable = True

        visited = []
        queue = []
        root_node_id = self.root_node.id

        visited.append(root_node_id)
        queue.append(root_node_id)

        while queue:
            node_id = queue.pop(0)
            node = self.get_node_info(node_id)
            for child in self.graph.successors(node_id):
      
                if child not in visited:
                    child_node = self.get_node_info(child)
                    child_node.unreachable = False
                    child_node.parent = node
                    child_node.action = self.get_edge_info(node, child_node).action
                    visited.append(child)
                    queue.append(child)

    def draw_graph(self):

        nodes_info = nx.get_node_attributes(self.graph, 'info')
        node_color_map = []
        node_size_map = []
        value_map = {}


        for node in nodes_info.values():

            node_size_map.append(30)

            if node == self.root_node:
                node_color_map.append('blue')

            elif node.chosen:
                node_color_map.append('lightblue')

            elif node.unreachable:
                node_color_map.append('grey')

            elif node in self.frontier and node not in self.new_nodes:
                node_color_map.append('green')

            elif node in self.new_nodes:
                node_color_map.append('pink')
                
            else:
                node_color_map.append('black')


        edges_info = nx.get_edge_attributes(self.graph, 'info')
        edge_color_map = []
        edge_width_map = []
        for edge in edges_info.values():
            if edge.node_from == self.root_node:
                edge_width_map.append(1)
                edge_color_map.append('blue')
          
            elif edge.node_to.chosen and edge.node_from.chosen:
                edge_width_map.append(1)
                edge_color_map.append('lightblue')
         
            else:
                edge_width_map.append(0.2)
                edge_color_map.append('grey')

        self.new_nodes.clear()

        general_options = {
            "with_labels": False,
            "font_size": 15,
        }

        node_options = {
            "node_color": node_color_map,
            "node_size": node_size_map,
        }

        edge_options = {
            "edge_color": edge_color_map,
            "width": edge_width_map,
            "arrowsize": 10,
        }

        H = nx.convert_node_labels_to_integers(self.graph, label_attribute='info')
        H_layout = pygraphviz_layout(H, prog="neato")
        G_layout = {H.nodes[n]['info']: p for n, p in H_layout.items()}

        options = {}
        options.update(general_options)
        options.update(node_options)
        options.update(edge_options)

        dpi = 96
        plt.figure(1, figsize=(1024/dpi, 768/dpi))
        nx.draw_networkx(self.graph, G_layout, **options)
        nx.draw_networkx_labels(self.graph, G_layout, value_map, font_size=8)
        handles = [Line2D([], [], color=color, label=label, marker='o')
           for color, label in zip([  "blue",     "lightblue",        "grey",           "green",       "pink",], 
                                   ["root node", "chosen node", "unreachable node", "frontier node", "new node"])]

        plt.legend(handles=handles)
        plt.show()


    def load_graph(self, path):
        self.graph = nx.readwrite.read_gpickle(path)

    def save_graph(self, path):
        nx.readwrite.write_gpickle(self.graph, path + ".gpickle")

    def write_graph(self, graph_object, path):
        nx.write_graphml(graph_object, path)
        
