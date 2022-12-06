import networkx as nx
import numpy as np
import os
import networkit as nk
import sys
sys.path.append("..")
from NetworkComponent.NetworkUtils import *
from NetworkComponent.NetworkCreatorNetworkit import *
import matplotlib.pyplot as plt
import pickle
from os import listdir
from os.path import isfile, join
from SimulationComponent.Simulation import Simulation


if __name__ == "__main__":
    for i in range(1):
        WA_ZIPCODE_COORDINATES_PATH = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'wa_zipcode_coordinates.csv'
        )
        print( WA_ZIPCODE_COORDINATES_PATH)
        M_PATH = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'M.npy'
        )

        empty_graph = nk.graph.Graph(weighted=False, directed=False)
        attribute_dict = {
            "income": NetworkUtils.generate_income_with_prob_value_list,
            "adoption": 0,
            "zipcode": 0,
            "degree": 0,
            "num_neighbor_adopted": 0,
            "adoption_time": -1
        }
        attribute_type_dict ={
            "income": float,
            "adoption": int,
            "zipcode": int,
            "degree": int,
            "num_neighbor_adopted": int,
            "adoption_time": int
        }
        path = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'wa_pev_weekly_reg.csv'
        )

        G = NetworkCreatorNetworkit(30000)
        G.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
        csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
        G.generate_nodes_from_population_income_csv(csv_path=csv_path)
        G.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
        G.generate_edges()
        G.set_node_degree()

        simulation_paras = {"income_coeff": 9.1e-6,
                            "neighbor_adoption_coeff": 8.62e-3}
        #860 5%
        simulation_time_length = 611
        simulation = Simulation(G, simulation_time_length, simulation_paras)
        print(simulation.G.scale)
        simulation.run()
        # simulation.output_weekly_cumulative_adoption()
        #simulation.output_cumulative_sum_by_zipcode(simulation_time_length)
        # simulation.show_adoption_history(path)
        # print("absolute error", simulation.calculate_absolute_error(path))
        # simulation.output_cumulative_sum_by_zipcode(simulation_time_length)
        simulation.output_heatmap()
