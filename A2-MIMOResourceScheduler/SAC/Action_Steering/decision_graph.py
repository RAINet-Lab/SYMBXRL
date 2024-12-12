'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import pandas as pd
import numpy as np
import networkx as nx
import pyvis.network as network

class DecisionGraph:
    def __init__(self, column_name) -> None:
        self.column_name = column_name
        self.decision_df = []
        self.G = nx.DiGraph()
        self.net = None
        self.previous_decision = None
        return

    def update_graph(self, symbolic_form_df: pd.DataFrame):
        """
        This function will receive a new decision and update the G graph object for the decision and the decisions count
        """
        current_decision = symbolic_form_df[self.column_name].iloc[0]
        current_reward = symbolic_form_df['reward'].iloc[0]
        
        # Add the current decision as a new node if it doesn't exist
        if current_decision not in self.G.nodes:
            self.G.add_node(current_decision, title=current_decision, occurrence=0, total_reward=0, mean_reward=0)
        
        # Update the occurrence count and total reward for the current decision node
        self.G.nodes[current_decision]['occurrence'] += 1
        self.G.nodes[current_decision]['total_reward'] += current_reward
        self.G.nodes[current_decision]['mean_reward'] = self.G.nodes[current_decision]['total_reward'] / self.G.nodes[current_decision]['occurrence']
        
        # Update the transition from the previous decision to the current decision
        if self.previous_decision is not None:
            if self.G.has_edge(self.previous_decision, current_decision):
                self.G[self.previous_decision][current_decision]['occurrence'] += 1
                self.G[self.previous_decision][current_decision]['total_reward'] += current_reward
            else:
                self.G.add_edge(self.previous_decision, current_decision, occurrence=1, total_reward=current_reward)
            
            self.G[self.previous_decision][current_decision]['mean_reward'] = self.G[self.previous_decision][current_decision]['total_reward'] / self.G[self.previous_decision][current_decision]['occurrence']
        
        self.previous_decision = current_decision
        self.previous_reward = current_reward  # Update the previous reward
        
        # Update probabilities and sizes for nodes and edges
        self._update_probabilities_and_sizes()
        
        return
    
    def _update_probabilities_and_sizes(self):
        """
        This function updates the probabilities and sizes of nodes and edges in the graph.
        """
        # Calculate the total occurrence count for nodes
        total_node_occurrence = sum(nx.get_node_attributes(self.G, 'occurrence').values())
        
        # Calculate the probability for each decision node
        node_probabilities = {}
        node_sizes = {}
        for node, data in self.G.nodes(data=True):
            node_probabilities[node] = data['occurrence'] / total_node_occurrence
            # Apply exponential scaling for node sizes
            min_size = 10
            scale_factor = 0.5  # Adjust this value to control the exponential growth rate of the sizes
            node_sizes[node] = min_size + np.exp(scale_factor * np.log1p(data['occurrence'] - 1))
        
        # Set the node probabilities and sizes in the graph
        nx.set_node_attributes(self.G, node_probabilities, 'probability')
        nx.set_node_attributes(self.G, node_sizes, 'size')
        
        # Calculate the probability for each edge
        edge_probabilities = {}
        for u, v, data in self.G.edges(data=True):
            total_transitions_from_u = sum(self.G[u][nbr]['occurrence'] for nbr in self.G.successors(u))
            edge_probabilities[(u, v)] = data['occurrence'] / total_transitions_from_u if total_transitions_from_u > 0 else 0
        
        # Set the edge probabilities in the graph
        nx.set_edge_attributes(self.G, edge_probabilities, 'probability')
    
    def build_graph(self):
        """
        This function will return the pyvis object that can be used to plot the created graph.
        """
        # Ensure probabilities and sizes are up to date
        self._update_probabilities_and_sizes()
        
        self.net = network.Network(height="1500px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=True, filter_menu=True, select_menu=True, cdn_resources="in_line")
        # Create the Pyvis network
        self.net.from_nx(self.G)
        
        # Iterate over each node in the Pyvis network to set the title with occurrence, mean reward, and probability
        for node in self.net.nodes:
            occurrence = self.G.nodes[node['id']]['occurrence']
            probability = round(100 * self.G.nodes[node['id']]['probability'], 1)
            mean_reward = self.G.nodes[node['id']]['mean_reward']
            node['title'] = f"Node: {node['id']} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"
        
        # Iterate over each edge in the Pyvis network to set the title with occurrence, mean reward, and probability
        for edge in self.net.edges:
            u, v = edge['from'], edge['to']
            occurrence = self.G[u][v]['occurrence']
            probability = round(100 * self.G[u][v]['probability'], 1)
            mean_reward = self.G[u][v]['mean_reward']
            edge['title'] = f"Edge from {u} to {v} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"
        
        # Calculate the graph size
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        
        # Create a text element for the graph size information
        size_text = f"Number of Nodes: {num_nodes}<br>Number of Edges: {num_edges}"
        
        # Add the size information as an HTML element to the Pyvis network
        self.net.add_node("size_info", label=size_text, shape="text", x='-95%', y=0, physics=False)
        
        self.net.barnes_hut(overlap=1)
        self.net.show_buttons(filter_=['physics'])
        return
    
    def get_graph(self, mode="all"):
        """
        This function will return the networkX object of the graph in order to perform analysis "Action Steering"
        """
        if mode == "all":
            return self.G, self.net
        
        if mode == "networkX":
            return self.G
        
        if mode == "pyvis":
            return self.net