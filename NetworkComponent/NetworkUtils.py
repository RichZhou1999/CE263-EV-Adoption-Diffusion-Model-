import numpy as np
import bisect
import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle
from typing import Union
from tqdm import tqdm
import itertools as it
import networkx as nx
import networkit as nk


def generate_age_value(*args,**kwargs):
    return np.random.randint(0, 80)


def generate_income_value(*args,**kwargs):
    return np.random.normal(10000, 5000)


def generate_zipcode_value(*args,**kwargs):
    return 94707


def generate_income_with_prob_value_list(prob_list=[], value_list=[], *args, **kwargs):
    if len(prob_list) !=len(value_list):
        raise('probality and value not ')
    sum_prob_list = []
    temp_sum = 0
    for prob in prob_list:
        temp_sum += prob
        sum_prob_list.append(temp_sum)
    r = np.random.random()
    index = bisect.bisect(sum_prob_list, r)
    if index < len(sum_prob_list):
        return value_list[index]
    else:
        return value_list[-1]


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

    if r == 0:
        p = 1
    else:
        p = r**-1.2 + 5e-6

    return p