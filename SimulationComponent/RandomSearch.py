import os
from SimulationComponent.Simulation import Simulation
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
import random
from pathlib import Path

def experiment(income_coeff, neighbor_adoption_coeff ):
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
    attribute_type_dict = {
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
    # simulation_paras = {"income_coeff": 9.75e-6,
    #                     "neighbor_adoption_coeff": 7.75e-3}
    simulation_paras = {"income_coeff": income_coeff,
                        "neighbor_adoption_coeff": neighbor_adoption_coeff}
    #
    simulation = Simulation(G, 700, simulation_paras)
    simulation.run()
    path = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'wa_pev_weekly_reg.csv'
    )
    absolute_error = simulation.calculate_absolute_error(path)
    print("absolute error", absolute_error)
    return absolute_error

possible_income_coeff = np.linspace(9.5, 10, 50)*1e-6
possible_neighbor_adoption_coeff = np.linspace(7.6, 7.9, 50)*1e-3
for i in range(1000):
    income_coeff = random.choice(possible_income_coeff)
    neighbor_adoption_coeff = random.choice(possible_neighbor_adoption_coeff)
    absolute_error = experiment(income_coeff, neighbor_adoption_coeff)
    my_file = Path("best_coeff.pkl")
    if not my_file.is_file():
        temp = {"income_coeff": income_coeff,
                "neighbor_adoption_coeff": neighbor_adoption_coeff,
                "absolute_error": absolute_error}
        with open("best_coeff.pkl", "wb") as f:
            pickle.dump(temp, f)
            print("initialize coeff")

    with open("best_coeff.pkl", "rb") as f:
        previous_result = pickle.load(f)
    if previous_result['absolute_error'] > absolute_error:
        temp = {"income_coeff":income_coeff,
                "neighbor_adoption_coeff": neighbor_adoption_coeff,
                "absolute_error": absolute_error}
        with open("best_coeff.pkl", "wb") as f:
            pickle.dump(temp, f)
            print("get better coeff")


with open("best_coeff.pkl","rb") as f:
    a = pickle.load(f)
    print(a)
# with open("best_coeff", "wb") as f:
#     pickle.dump(a, f)
#
# with open("best_coeff", "rb") as f:
#     a = pickle.load(f)
