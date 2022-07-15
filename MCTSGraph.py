import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import pygraphviz_layout
from matplotlib.lines import Line2D

class Graph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_node = None
        self.new_nodes = []

    def add_node(self, node):
        self.graph.add_node(node.id, info=node)
        self.new_nodes.append(node)

    def add_edge(self, edge):
        self.graph.add_edge(edge.node_from.id, edge.node_to.id, info=edge)

    def has_observation(self, obs):
        nodes_info = nx.get_node_attributes(self.graph, 'info')
        for node in nodes_info.values():
            if node.observation == obs:
                return True 
        return False 

    def get_children(self, node):
        node_list = []
        for n in self.graph.successors(node.id):
            child_node = self.graph.nodes[n]['info']
            node_list.append(child_node)

        return node_list

    def get_edge_info(self, parent, child):
        return self.graph.get_edge_data(parent.id, child.id)['info']


    def draw_graph(self):
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
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

            elif node.redundant:
                node_color_map.append('grey')

            elif node.is_leaf and node not in self.new_nodes:
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
           for color, label in zip([  "blue",     "lightblue",       "grey",        "green",     "pink",],
                                   ["root node", "chosen node", "redundant node", "leaf node", "new node",])]

        plt.legend(handles=handles)
        plt.show()


    def load_graph(self, path):
        self.graph = nx.readwrite.read_gpickle(path)

    def save_graph(self, path):
        nx.readwrite.write_gpickle(self.graph, path + ".gpickle")

    def write_graph(self, graph_object, path):
        nx.write_graphml(graph_object, path)
        
