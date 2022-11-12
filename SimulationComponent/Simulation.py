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
        self.income_coefficient = simulation_paras["income_coefficient"]
        self.friendship_adoption_coefficient = simulation_paras["friendship_adoption_coefficient"]
        self.current_adoption_number = 0
        self.adoption_history_list = []

    def run(self):
        for i in range(self.simulation_time_length):
            self.step()
        print(self.adoption_history_list)

    def step(self):
        self.transition()

    def transition(self):
        for node_id in range(self.G.current_node_number):
            threshold = np.random.random()
            temp_income = self.G.node_attributes_attachment['income'][node_id]
            p = self.income_coefficient * temp_income
            if p > threshold and self.G.node_attributes_attachment['adoption'][node_id] == 0:
                self.G.node_attributes_attachment['adoption'][node_id] = 1
                self.current_adoption_number += 1
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
        "income": generate_income_with_prob_value_list,
        "adoption": 0,
        "zip code": 0,
    }
    attribute_type_dict ={"income": float,
                          "adoption": int,
                          "zip code": int,}

    graph = NetworkCreatorNetworkit(10000)
    graph.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    graph.generate_nodes_from_population_income_csv(csv_path=csv_path)
    print("zip code for node 3000:", graph.node_attributes_attachment['zip code'][3000])
    print("income for node 3000:", graph.node_attributes_attachment['income'][3000])
    print("adoption for node 3000:", graph.node_attributes_attachment['adoption'][3000])

    graph.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
    graph.generate_edges()

    # plot the graph and save figure
    # nk.viztasks.drawGraph(graph.G)
    # plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'G.png'), dpi=300, bbox_inches='tight')

    print(f"number of nodes = {graph.numberOfNodes()}")
    print(f"number of edges = {graph.numberOfEdges()}")
    simulation_paras = {"income_coefficient": 1e-3,
                        "friendship_adoption_coefficient": 0.001}
    simulation = Simulation(graph, 500, simulation_paras)
    simulation.run()
    simulation.show_adoption_history()
