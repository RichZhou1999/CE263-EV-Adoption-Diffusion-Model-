import networkit
import pandas as pd
import typing
import numpy as np
import math
import copy
from NetworkUtils import generate_income_with_prob_value_list
import os

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


class NetworkCreatorNetworkit:
    def __init__(self, graph, node_attribute_dict, designed_node_number=10000):
        self.G = graph
        self.current_node_number = 0
        self.node_attributes_attachment = {}
        for key in node_attribute_dict:
            self.node_attributes_attachment[key] = self.G.attachNodeAttribute(key, int)
        self.node_attribute_dict = node_attribute_dict
        self.designed_node_number = designed_node_number
        self.scale = None

    def set_scale_value(self, true_value, model_value):
        self.scale = model_value/true_value

    def generate_nodes(self, number: int, attribute_dict, **kwargs):
        for i in range(self.current_node_number, self.current_node_number + number):
            node_id = i
            self.G.addNode()
            for key, function in attribute_dict.items():
                if callable(function):
                    value = function(**kwargs)
                else:
                    value = function
                self.node_attributes_attachment[key][node_id] = value
        self.current_node_number += number

    def generate_nodes_from_population_income_csv(self, csv_path,):
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
            prob_list = list(item["<10,000":">200,000"]/100)
            temp_attribute_dict = copy.copy(self.node_attribute_dict)
            temp_attribute_dict.update({"zip code": int(item['zip code'])})
            self.generate_nodes(number, temp_attribute_dict, prob_list=prob_list, value_list=value_list)


if __name__ == "__main__":
    empty_graph = networkit.graph.Graph(weighted=False, directed=False)
    attribute_dict = {
        "income": generate_income_with_prob_value_list,
        "adoption": 0,
        "zip code": 0,
    }
    graph = NetworkCreatorNetworkit(empty_graph, attribute_dict)
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
    graph.generate_nodes_from_population_income_csv(csv_path=csv_path)
    print("zip code for node 3000:",graph.node_attributes_attachment['zip code'][3000])
    print("income for node 3000:",graph.node_attributes_attachment['income'][3000])
    print("adoption for node 3000:", graph.node_attributes_attachment['adoption'][3000])


