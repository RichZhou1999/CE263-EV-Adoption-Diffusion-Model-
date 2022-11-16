import networkx as nx
import numpy as np
import os
import networkit as nk
import sys
sys.path.append("..")
from NetworkComponent.NetworkUtils import *
from NetworkComponent.NetworkCreatorNetworkit import *
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, graph,
                 simulation_time_length: int, simulation_paras: dict):
        self.G = graph
        self.simulation_time_length = simulation_time_length
        self.current_simulation_time = 0
        self.income_coeff = simulation_paras["income_coeff"]
        self.neighbor_adoption_coeff = simulation_paras["neighbor_adoption_coeff"]
        self.current_adoption_number = 0
        self.adoption_history_list = []

    def run(self):
        
        print()
        print("========================")
        print("Diffusion simulation starting...")

        for i in tqdm(range(self.simulation_time_length)):
            self.step()
        
        print(self.adoption_history_list)
        print("DONE")
        print()

    def step(self):
        self.transition()
        # updates number of adopted neighbors of each node
        # self.G.update_num_neighbor_adopted()
    def transition(self):
        for node_id in range(self.G.current_node_number):

            adopt_thr = np.random.random()
            agent_income = self.G.node_attributes_attachment['income'][node_id]
            agent_num_neighbor_adopted = self.G.node_attributes_attachment['num_neighbor_adopted'][node_id]
            p = self.income_coeff * agent_income + self.neighbor_adoption_coeff * agent_num_neighbor_adopted

            if p > adopt_thr and self.G.node_attributes_attachment['adoption'][node_id] == 0:
                self.G.node_attributes_attachment['adoption'][node_id] = 1
                self.current_adoption_number += 1
                for neighbor in self.G.iterNeighbors(node_id):
                    self.G.node_attributes_attachment['num_neighbor_adopted'][neighbor] += 1

        self.adoption_history_list.append(self.current_adoption_number)

    def show_adoption_history(self):
        plt.figure()
        y = np.array(self.adoption_history_list) / self.G.scale
        x = range(len(self.adoption_history_list))
        plt.plot(x, y)
        plt.xlabel('Step')
        plt.ylabel("Adoption Number")
        plt.show()

if __name__ == "__main__":

    WA_ZIPCODE_COORDINATES_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'wa_zipcode_coordinates.csv'
    )
    M_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'M.npy'
    )

    empty_graph = nk.graph.Graph(weighted=False, directed=False)
    attribute_dict = {
        "income": NetworkUtils.generate_income_with_prob_value_list,
        "adoption": 0,
        "zipcode": 0,
        "degree": 0,
        "num_neighbor_adopted": 0
    }
    attribute_type_dict ={
        "income": float,
        "adoption": int,
        "zipcode": int,
        "degree": int,
        "num_neighbor_adopted": int,
    }

    # network initialization
    G = NetworkCreatorNetworkit(25000)
    G.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    G.generate_nodes_from_population_income_csv(csv_path=csv_path)
    G.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
    G.generate_edges()
    G.set_node_degree()

    print("zip code for node 3000:", G.node_attributes_attachment['zipcode'][3000])
    print("income for node 3000:", G.node_attributes_attachment['income'][3000])
    print("adoption for node 3000:", G.node_attributes_attachment['adoption'][3000])
    print("degree for node 3000:", G.node_attributes_attachment['degree'][3000])

    print(f"number of nodes = {G.numberOfNodes()}")
    print(f"number of edges = {G.numberOfEdges()}")

    simulation_paras = {"income_coeff": 2e-5,
                        "neighbor_adoption_coeff": 8e-6}
    
    simulation = Simulation(G, 600, simulation_paras)
    simulation.run()
    simulation.show_adoption_history()
