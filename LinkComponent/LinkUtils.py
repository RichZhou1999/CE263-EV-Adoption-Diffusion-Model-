import csv
import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle
from typing import Union
from tqdm import tqdm
import itertools as it
import networkx as nx
import snap

def generate_inter_zipcode_dist_matrix(path, mode='npy') -> Union[np.ndarray, None]:
    """
    generate inter zipcode great circle distance as 2d matrix
    only intended to be ran once as lookup matrix

    Parameters:
        path (str): path to wa_zipcode_coord.csv
        mode (str): save to .npy file or return as numpy array
    Returns:
        M (np.ndarray): 2d symmetric matrix with 0s on the diagonal
        None: if mode='npy'
    """

    wa_zipcode_coord = pd.read_csv(path)
    wa_zipcode_coord = wa_zipcode_coord.reset_index()
    num_nzip = wa_zipcode_coord['Zip'].nunique()

    M = np.zeros((num_nzip, num_nzip))

    for idx_start in tqdm(np.arange(M.shape[0])):
        for idx_end in np.arange(M.shape[1]):
            if idx_start == idx_end:
                M[idx_start][idx_end] = 0
            else:
                start = (
                    wa_zipcode_coord[wa_zipcode_coord['index'] == idx_start]['Latitude'].iloc[0], 
                    wa_zipcode_coord[wa_zipcode_coord['index'] == idx_start]['Longitude'].iloc[0]
                )

                end = (
                    wa_zipcode_coord[wa_zipcode_coord['index'] == idx_end]['Latitude'].iloc[0], 
                    wa_zipcode_coord[wa_zipcode_coord['index'] == idx_end]['Longitude'].iloc[0]
                )

                M[idx_start][idx_end] = great_circle(start, end).kilometers
    
    if mode == 'npy':
        file_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'M.npy'))
        np.save(file_path, M)
    else:
        return M

def link_connect_prob(r: float) -> float:
    """
    return the probability of link connection between two nodes separated by r in great circle distance
    
    Parameters:
        r (float): great circle distance between centroids of two zip codes
    Returns: 
        p (float): probability
    """

    assert r >= 0, 'r needs to be positive'
    p = r**-1.2 + 5e-6

    return p


def generate_edge_df(path: str, path_M: str, G: nx.Graph) -> pd.DataFrame:
    """
    generate df to be added as edges, [source, target, distance]

    Parameters:
        path (str): path to wa_zipcode_coord.csv
        path_M (str): path to M numpy array
        G (nx.Graph): networkx graph object with nodes added
    Returns:
        edges_df (pd.Dataframe): edge dataframe

    """

    node_list = list(G.nodes)
    node_comb = list(it.combinations(node_list, 2))

    wa_zipcode_coord = pd.read_csv(path)
    M = np.load(path_M)

    zipcode_idx_dict = wa_zipcode_coord.reset_index().set_index('Zip')['index'].to_dict()
    edge_df_data = []

    for node_tuple in node_comb:
        
        node_start = node_tuple[0]
        node_end = node_tuple[1]

        zipcode_start = int(G.nodes[node_start]['zipcode'])
        zipcode_end = int(G.nodes[node_end]['zipcode'])

        idx_start = zipcode_idx_dict[zipcode_start]
        idx_end = zipcode_idx_dict[zipcode_end]

        r = M[idx_start][idx_end]
        p = np.random.random(1)[0]
        p_connect = link_connect_prob(r)

        if p >= p_connect:
            edge_df_data.append([node_start, node_end, r])

    edges_df = pd.DataFrame(edge_df_data, columns=['source', 'target', 'distance'])

    return edges_df

def add_edges_snap(G, edges_df):

    
    
    return

"""

if __name__ == "__main__":

    generate_inter_zipcode_dist_matrix()

"""