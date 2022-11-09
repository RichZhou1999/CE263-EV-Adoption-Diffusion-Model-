import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Callable
from NodeUtils import generate_income_with_prob_value_list
import pandas as pd
import copy
import math
'''
NodeCreator generator customized node according to the need.
Parameter:
    self._nodes: store the nodes
    self.attribute_dict: define the attributes needed for the node
    it is a dictionary with name as the key, and callable function as 
    value.
    If the value of the dictionary is not callable then the value is set fixed. eg: zipcode
Method:
    generate_nodes(number, attribute_dict): number define the number of nodes needed
    This method can be used several times with all the previous nodes remained. 
    generate_nodes_from_population_income_csv(csv_file, attribute_dict, number_of_simulation_nodes): simulate the distribution of income and
    create model agents according to the population distribution
'''


class NodeCreator:
    def __init__(self):
        self._nodes: list[set[int, dict]] = []

    def generate_nodes(self, number: int, attribute_dict: dict[str, callable], **kwargs):
        for i in range(len(self.nodes), len(self.nodes) + number):
            node_id = i
            node_attribute = {}
            for key, function in attribute_dict.items():
                if callable(function):
                    value = function(**kwargs)
                else:
                    value = function
                node_attribute.update({key: value})
            self._nodes.append((node_id, node_attribute))

    def generate_nodes_from_population_income_csv(self, csv_path, attribute_dict,  number_of_simulation_nodes=100000):
        data = pd.read_csv("%s.csv" % csv_path)
        population_sum = sum(data['Population'])
        scale = number_of_simulation_nodes / population_sum
        print("model agents number: ", number_of_simulation_nodes)
        print("data agents number: ", population_sum)
        print("scale:", scale)
        for i in range(len(data)):
            item = data.iloc[i]
            number = math.ceil(item['Population'] * scale)
            value_list = [5000, 12500, 20000, 30000, 42750, 67500,
                          87500, 125000, 175000, 200000]
            prob_list = list(item["<10,000":">200,000"]/100)
            temp_attribute_dict = copy.copy(attribute_dict)
            temp_attribute_dict.update({"zip code": item['zip code']})
            self.generate_nodes(number, temp_attribute_dict, prob_list=prob_list, value_list=value_list)

    @property
    def nodes(self) -> list[set[int, dict]]:
        return self._nodes


if __name__ == "__main__":
    import networkx as nx
    G = nx.Graph()

    attribute_dict = {
        "income": generate_income_with_prob_value_list,
        "Adoption": False
    }

    nodeCreateor = NodeCreator()
    nodeCreateor.generate_nodes_from_population_income_csv(csv_path='.././Data/BEV_data', attribute_dict=attribute_dict)

    G.add_nodes_from(nodeCreateor.nodes)
    print(G.nodes)
    print("node_1000:", G.nodes[1000])
    print("node_5000:", G.nodes[5000])
