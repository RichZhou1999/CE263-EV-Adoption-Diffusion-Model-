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
from SimulationComponent.Simulation import Simulation

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

    # network initialization
    # G = NetworkCreatorNetworkit(30000)
    # G.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
    # csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    # G.generate_nodes_from_population_income_csv(csv_path=csv_path)
    # G.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
    # G.generate_edges()
    # G.set_node_degree()
    #
    # print("zip code for node 3000:", G.node_attributes_attachment['zipcode'][3000])
    # print("income for node 3000:", G.node_attributes_attachment['income'][3000])
    # print("adoption for node 3000:", G.node_attributes_attachment['adoption'][3000])
    # print("degree for node 3000:", G.node_attributes_attachment['degree'][3000])
    #
    # print(f"number of nodes = {G.numberOfNodes()}")
    # print(f"number of edges = {G.numberOfEdges()}")

    # simulation_paras = {"income_coeff": 9.5e-6,
    #                     "neighbor_adoption_coeff": 7.75e-3}
    # with open("best_coeff.pkl","rb") as f:
    #     best_coeff = pickle.load(f)
        # print(best_coeff)
    # simulation_paras = {"income_coeff": 9.959e-6,
    #                     "neighbor_adoption_coeff": 0.0076}
    #
    # simulation = Simulation(G, 690, simulation_paras)
    # simulation.run()
    path = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'wa_pev_weekly_reg.csv'
    )
    # print("absolute error", simulation.calculate_absolute_error(path))
    # simulation.show_adoption_history(path)



    G = NetworkCreatorNetworkit(30000)
    G.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    G.generate_nodes_from_population_income_csv(csv_path=csv_path)
    G.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
    G.generate_edges()
    G.set_node_degree()
    #9.1e-6,8.62e-3
    #9.3 8.426
    #9.23, 8.48
    simulation_paras = {"income_coeff": 4e-5,
                        "neighbor_adoption_coeff": 0}
    simulation_time_length = 612
    simulation = Simulation(G, simulation_time_length, simulation_paras)
    simulation.run()
    print("absolute error", simulation.calculate_absolute_error(path))
    simulation.show_adoption_history(path)
    simulation.output_adoption_history()
    # ZIPCODE_COORDINATES = pd.read_csv(WA_ZIPCODE_COORDINATES_PATH)
    # city = "Tacoma"
    # print(ZIPCODE_COORDINATES['City'])
    # zipcode_list = ZIPCODE_COORDINATES[ZIPCODE_COORDINATES['City'] ==city]['Zip']
    # zipcode_list = list(zipcode_list)
    # print(zipcode_list)
    #
    # simulation.plot_zipcode_adoption_curve(zipcode_list, city)
