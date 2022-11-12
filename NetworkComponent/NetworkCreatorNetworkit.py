import sys
sys.path.append("..")
import networkit as nk
import pandas as pd
import typing
import numpy as np
import math
import copy
import NetworkComponent.NetworkUtils as NetworkUtils
# from NetworkUtils import *
import os
import itertools as it
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(0)

'''
Create network with the package of Networkit:
Node:
    ----------------------------------------------------------------
    Nodes inside the graph are created with the function of add_node():
    The index of the node starts from 0. When add_node() is called once, the index 
    increases by one.
    The attributes of the nodes are stored in "node_attributes_attachment", which is separated
    with the definition of the nodes acting like an attachment above the nodes:
    "node_attributes_attachment" is a double dict: dict[dict[]], the first key
    is the name of the attribute, the second key is the node_id.
    node_id ranges from 0 to the number of designed nodes number, eg:100000
Link:
    ----------------------------------------------------------------

'''


class NetworkCreatorNetworkit(nk.graph.Graph):
    # def __init__(self,**kwargs):
    #     super(NetworkCreatorNetworkit, self).__init__()
    def __init__(self, designed_node_number):
        super().__init__()
        self.current_node_number = 0
        self.node_attributes_attachment = {}
        self.designed_node_number = designed_node_number
        self.scale = None
        self.edge_list = []
        self.node_attribute_dict = {}

    def generate_node_attribute_attachment(self, node_attribute_dict,attribute_type_dict):
        self.node_attribute_dict = node_attribute_dict
        for key in node_attribute_dict:
            self.node_attributes_attachment[key] = self.attachNodeAttribute(key, attribute_type_dict[key])

    def set_scale_value(self, true_value, model_value):
        self.scale = model_value/true_value

    def generate_nodes(self, number: int, attribute_dict, **kwargs):
        for i in range(self.current_node_number, self.current_node_number + number):
            node_id = i
            self.addNode()
            for key, function in attribute_dict.items():
                if callable(function):
                    value = function(**kwargs)
                else:
                    value = function
                self.node_attributes_attachment[key][node_id] = value
        self.current_node_number += number

    def generate_nodes_from_population_income_csv(self, csv_path):
        # data = pd.read_csv("%s.csv" % csv_path)
        data = pd.read_csv(csv_path)
        population_sum = sum(data['Population'])
        self.set_scale_value(population_sum, self.designed_node_number)
        print("model agents number: ", self.designed_node_number)
        print("data agents number: ", population_sum)
        print("scale:", self.scale)
        for i in range(len(data)):
            item = data.iloc[i]
            number = math.ceil(item['Population'] * self.scale)
            value_list = [5000, 12500, 20000, 30000, 42750, 67500,
                          87500, 125000, 175000, 200000]
            value_list = np.array(value_list)/max(value_list)
            prob_list = list(item["<10,000":">200,000"]/100)
            temp_attribute_dict = copy.copy(self.node_attribute_dict)
            temp_attribute_dict.update({"zip code": int(item['zip code'])})
            self.generate_nodes(number, temp_attribute_dict, prob_list=prob_list, value_list=value_list)

    def generate_edge_list(self, path: str, path_M: str) -> None:
        """
        generate edges based on node's inter-zipcode distance

        Parameters:
            path (str): path to wa_zipcode_coord.csv
            path_M (str): path to M numpy array
        """
        print()
        print("========================")

        if os.path.isfile(
            os.path.join(os.path.dirname(__file__), '..', 'Data/edge_list.npy')
        ):  
            print('Data/edge_list.npy found...')
            self.edge_list = np.load(os.path.join(os.path.dirname(__file__), '..', 'Data/edge_list.npy'))
            print("DONE!")
            return

        node_list = list(range(self.G.numberOfNodes()))
        edge_comb = list(it.combinations(node_list, 2))
        wa_zipcode_coord = pd.read_csv(path)
        M = np.load(path_M)
        zipcode_idx_dict = wa_zipcode_coord.reset_index().set_index('Zip')['index'].to_dict()

        print('generating edge list...')
        def node_comb_filter(node_tuple: tuple) -> bool:

            node_start = node_tuple[0]
            node_end = node_tuple[1]

            # zipcode
            zipcode_start = int(self.node_attributes_attachment['zip code'][node_start])
            zipcode_end = int(self.node_attributes_attachment['zip code'][node_end])

            # zipcode idx
            idx_start = zipcode_idx_dict[zipcode_start]
            idx_end = zipcode_idx_dict[zipcode_end]

            r = M[idx_start][idx_end]
            p = np.random.random(1)[0]
            p_connect = NetworkUtils.link_connect_prob(r)

            return p < p_connect

        self.edge_list = list(filter(node_comb_filter, edge_comb))

        print(f"possible node combination = {len(edge_comb)}")
        print(f"number of link combination = {len(self.edge_list)}, \
            making up {len(self.edge_list) / len(edge_comb) * 100}% of all combination")

        print('saving edge list to npy..')
        np.save(os.path.join(os.path.dirname(__file__), '..', 'Data/edge_list.npy'), np.array(self.edge_list))
        print("DONE!")

    def generate_edges(self) -> None:
        """
        adding edges to networkit graph
        """

        print()
        print('generating edges...')

        for row in tqdm(self.edge_list):
            source = row[0]
            target = row[1]
            self.addEdge(source, target)
        
        print("DONE!")

if __name__ == "__main__":

    WA_ZIPCODE_COORDINATES_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'wa_zipcode_coordinates.csv'
    )
    M_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'Data', 'M.npy'
    )

    # empty_graph = nk.graph.Graph(weighted=False, directed=False)
    attribute_dict = {
        "income": NetworkUtils.generate_income_with_prob_value_list,
        "adoption": 0,
        "zip code": 0,
    }
    attribute_type_dict ={"income": float,
                          "adoption": int,
                          "zip code": int,}
    # graph = NetworkCreatorNetworkit(1)
    graph = NetworkCreatorNetworkit(10000)
    graph.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    graph.generate_nodes_from_population_income_csv(csv_path=csv_path)
    print("zip code for node 3000:", graph.node_attributes_attachment['zip code'][3000])
    print("income for node 3000:", graph.node_attributes_attachment['income'][3000])
    print("adoption for node 3000:", graph.node_attributes_attachment['adoption'][3000])
    #
    graph.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
    graph.generate_edges()

    # plot the graph and save figure
    # nk.viztasks.drawGraph(graph.G)
    # plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'G.png'), dpi=300, bbox_inches='tight')

    print(f"number of nodes = {graph.numberOfNodes()}")
    print(f"number of edges = {graph.numberOfEdges()}")







    # class NetworkCreatorNetworkit(nk.graph.Graph):
    #     def __init__(self, t):
    #         super(NetworkCreatorNetworkit, self).__init__()
    #         self.t = 123

    # a = NetworkCreatorNetworkit(123)
    # print(a.numberOfNodes())

