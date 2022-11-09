import matplotlib.pyplot as plt
from NodeComponent.NodeCreator import *
from NodeComponent.NodeUtils import *
from LinkComponent.LinkUtils import *
import networkx as nx

# G = nx.Graph()
# attribute_dict = {
#     "age": value_generator_dict['age'],
#     "income": value_generator_dict['income'],
#     "zipcode": value_generator_dict['zipcode']
# }
#
# nodeCreator = NodeCreator(attribute_dict)
# nodeCreator.generate_nodes(5)
# nodeCreator.generate_nodes(5)
# G.add_nodes_from(nodeCreator.nodes)
# import math
# print(int(math.ceil(0.5)))


# TODO: construct Experiment class to store settings variables and graph objects

WA_ZIPCODE_COORDINATES_PATH = os.path.join(
    os.path.dirname(__file__), 'Data', 'wa_zipcode_coordinates.csv'
)
M_PATH = os.path.join(
    os.path.dirname(__file__), 'Data', 'M.npy'
)

if __name__ == "__main__":

    G = nx.Graph()

    attribute_dict = {
        "income": generate_income_with_prob_value_list,
        "Adoption": False
    }

    nodeCreateor = NodeCreator()
    nodeCreateor.generate_nodes_from_population_income_csv(
        csv_path=os.path.realpath(os.path.join(os.path.dirname(__file__), 'Data', 'BEV_data.csv')), 
        attribute_dict=attribute_dict
    )

    G.add_nodes_from(nodeCreateor.nodes)
    edges_df = generate_edge_df(WA_ZIPCODE_COORDINATES_PATH, M_PATH, G)

    # 28.56G heap space for 10k agent
    G = nx.from_pandas_edgelist(edges_df, edge_attr='distance')
    nx.write_gpickle(
        G, 
        os.path.join(
            os.path.dirname(__file__), 'Data', 'G.gpickle'
        )
    )
    
    # <k>
    print(round(2 * G.number_of_edges() / G.number_of_nodes(), 3))
    nx.draw(G)
    plt.savefig("G.png")