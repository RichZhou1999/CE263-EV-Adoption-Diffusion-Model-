import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Callable
from NodeUtils import generate_age_value, generate_income_value, generate_zipcode_value

'''
NodeCreator generator customized node according to the need.
Parameter:
    self._nodes: store the nodes
    self.attribute_dict: define the attributes needed for the node
    it is a dictionary with name as the key, and callable function as 
    value
Method:
    generate_nodes(number): number define the number of nodes needed
    This method can be used several times with all the previous nodes remained. 
'''


class NodeCreator:
    def __init__(self, attribute_dict: dict[str, callable]):
        self._nodes: list[set[int, dict]] = []
        self.attribute_dict = attribute_dict

    def generate_nodes(self, number: int):
        for i in range(len(self.nodes), len(self.nodes) + number):
            node_id = i
            node_attribute = {}
            for key, function in self.attribute_dict.items():
                value = function()
                node_attribute.update({key: value})
            self._nodes.append((node_id, node_attribute))

    @property
    def nodes(self) -> list[set[int, dict]]:
        return self._nodes


if __name__ == "__main__":
    import networkx as nx
    G = nx.Graph()
    attribute_dict = {
        "age": generate_age_value,
        "income": generate_income_value,
        "zipcode": generate_zipcode_value
    }
    nodeCreateor = NodeCreator(attribute_dict)
    nodeCreateor.generate_nodes(5)
    nodeCreateor.generate_nodes(5)
    G.add_nodes_from(nodeCreateor.nodes)
    print(G.nodes)
    print(G.nodes[9])