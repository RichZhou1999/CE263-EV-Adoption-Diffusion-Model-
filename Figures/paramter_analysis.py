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

path = os.path.join(
    os.path.dirname(__file__), '..', 'Data', 'wa_pev_weekly_reg.csv'
)
empirical_data = pd.read_csv(path)
empirical_data = empirical_data['cum_reg']
with open("two_para_adoption_history.pkl", "rb") as f:
    two_parameter_data = pickle.load(f)
with open("income_adoption_history.pkl", "rb") as f:
    income_parameter_data = pickle.load(f)
with open("neighbor_adoption_history.pkl", "rb") as f:
    neighbor_parameter_data = pickle.load(f)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(10,5))
x = range(len(two_parameter_data))
plt.plot(x, two_parameter_data, label='income and neighbor #', linewidth=2,color=colors[0])

empirical_x = range(len(empirical_data))
plt.plot(empirical_x, empirical_data, label="empirical",linewidth=2,color=colors[1])
# plt.plot(x, income_parameter_data, label='income', linewidth=2, color=colors[2])
# plt.plot(x, neighbor_parameter_data, label='neighbor', linewidth=2, color=colors[3])
plt.xlabel('Time',fontsize=15)
plt.ylabel("Adoption Number",fontsize=15)

x_label_position = [0, 104, 208, 312, 416, 520, 624]
labels = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
plt.grid(False)
plt.xticks(x_label_position, labels)
plt.legend(fontsize=15)
plt.tick_params(labelsize=15)
# plt.savefig("two_para")
# plt.show()

def calculate_error(data1,data2):
    min_length = min(len(data1), len(data2))
    error = 0
    for i in range(min_length):
        error += abs(data1[i] - data2[i])
    print(error / min_length)
    return error / min_length


p = np.polyfit(range(len(empirical_data)), np.log(empirical_data), 1)
a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(0, len(empirical_data), 100)
y_fitted = a * np.exp(b * x_fitted)
calculate_error(empirical_data,y_fitted)
calculate_error(empirical_data,income_parameter_data)
calculate_error(empirical_data,neighbor_parameter_data)
calculate_error(empirical_data,two_parameter_data)

