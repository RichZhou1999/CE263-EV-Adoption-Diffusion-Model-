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
from os import listdir
from os.path import isfile, join


class Simulation:
    def __init__(self, graph,
                 simulation_time_length: int, simulation_paras: dict):
        self.G = graph
        self.simulation_time_length = simulation_time_length
        self.current_simulation_time = 0
        self.income_coeff = simulation_paras["income_coeff"]
        self.neighbor_adoption_coeff = simulation_paras["neighbor_adoption_coeff"]
        self.current_adoption_number = 0
        self.adoption_history_list = []

    def run(self):
        
        print()
        print("========================")
        print("Diffusion simulation starting...")

        for i in tqdm(range(self.simulation_time_length)):
            self.step()
            self.current_simulation_time += 1
        
        print(self.adoption_history_list)
        print("DONE")
        print()

    def step(self):
        self.transition()
        # updates number of adopted neighbors of each node
        # self.G.update_num_neighbor_adopted()
    def transition(self):
        for node_id in range(self.G.current_node_number):

            adopt_thr = np.random.random()
            agent_income = self.G.node_attributes_attachment['income'][node_id]
            agent_num_neighbor_adopted = self.G.node_attributes_attachment['num_neighbor_adopted'][node_id]
            p = self.income_coeff * agent_income + self.neighbor_adoption_coeff * agent_num_neighbor_adopted/self.G.max_degree

            if p > adopt_thr and self.G.node_attributes_attachment['adoption'][node_id] == 0:
                self.G.node_attributes_attachment['adoption'][node_id] = 1
                self.G.node_attributes_attachment["adoption_time"][node_id] = self.current_simulation_time
                self.current_adoption_number += 1
                for neighbor in self.G.iterNeighbors(node_id):
                    self.G.node_attributes_attachment['num_neighbor_adopted'][neighbor] += 1
        self.adoption_history_list.append(self.current_adoption_number)

    def show_adoption_history(self, empirical_csv_path):
        fig = plt.figure()
        y = np.array(self.adoption_history_list) / self.G.scale
        x = range(len(self.adoption_history_list))
        plt.plot(x, y, label='model curve', linestyle='--')
        empirical_data = pd.read_csv(empirical_csv_path)
        empirical_data = empirical_data['cum_reg']
        empirical_data_time_range = range(len(empirical_data))
        plt.plot(empirical_data_time_range, empirical_data, label='empirical curve', linestyle="solid")
        p = np.polyfit(range(len(empirical_data)), np.log(empirical_data), 1)
        a = np.exp(p[1])
        b = p[0]
        x_fitted = np.linspace(0, len(empirical_data), 100)
        y_fitted = a * np.exp(b * x_fitted)
        plt.plot(x_fitted, y_fitted, label='baseline exponential curve',linestyle="-.")
        plt.xlabel('Time')
        plt.ylabel("Adoption Number")

        x_label_position = [0, 104, 208, 312, 416, 520, 624]
        labels = [2011, 2013, 2015, 2017, 2019, 2021, 2022]
        plt.grid(False)
        plt.xticks(x_label_position, labels)
        # locs, labels = plt.xticks()
        # for label in labels:
        #     label.set_visible(False)
        # for label in labels[0::10]:
        #     label.set_visible(True)
        plt.legend()
        plt.show()

    def calculate_absolute_error(self, empirical_csv_path):
        empirical_data = pd.read_csv(empirical_csv_path)
        empirical_data = empirical_data['cum_reg']
        empirical_data_time_range = range(len(empirical_data))
        min_length = min(len(self.adoption_history_list), len(empirical_data_time_range))
        model_data = np.array(self.adoption_history_list) / self.G.scale
        error = 0
        for i in range(min_length):
            error += abs(empirical_data[i] - model_data[i])
            if i == min_length - 1:
                print(abs(empirical_data[i] - model_data[i]))
        return error

    def reset(self):
        self.adoption_history_list = []
        self.current_simulation_time = 0
        self.current_adoption_number = 0
        self.G.reset()

    def plot_zipcode_adoption_curve(self, zipcode_list,city):
        # adoption_each_step_dict = {}
        # for node_id in range(self.G.current_node_number):
        #     if self.G.node_attributes_attachment['adoption'][node_id] == 1:
        #         if self.G.node_attributes_attachment['zipcode'][node_id] in adoption_each_step_dict:
        #             adoption_each_step_dict[self.G.node_attributes_attachment['zipcode'][node_id]][self.G.node_attributes_attachment['adoption_time'][node_id]] += 1
        #         else:
        #             adoption_each_step_dict[self.G.node_attributes_attachment['zipcode']] = [0] * self.simulation_time_length
        #             adoption_each_step_dict[self.G.node_attributes_attachment['zipcode'][node_id]][self.G.node_attributes_attachment['adoption_time'][node_id]] += 1

        adoption_each_step = [0] * self.simulation_time_length

        for node_id in range(self.G.current_node_number):
            if (self.G.node_attributes_attachment['zipcode'][node_id] in zipcode_list) and self.G.node_attributes_attachment['adoption'][node_id] == 1:
                adoption_each_step[self.G.node_attributes_attachment['adoption_time'][node_id]] += 1
        temp_sum = 0
        adoption_sum_list = []
        for i in range(self.simulation_time_length):
            temp_sum += adoption_each_step[i]
            adoption_sum_list.append(temp_sum)
        y = np.array(adoption_sum_list)/ self.G.scale
        x = range(len(adoption_sum_list))
        plt.plot(x, y, label='model curve', linestyle='--')
        x_label_position = [0, 104, 208, 312, 416, 520, 624]
        labels = [2011, 2013, 2015, 2017, 2019, 2021, 2022]
        plt.grid(False)
        plt.xticks(x_label_position, labels)
        plt.legend()
        plt.title("adoption curve under %s"%city)
        plt.show()

    def output_weekly_cumulative_adoption(self):
        adoption_each_step = [0] * self.simulation_time_length

        for node_id in range(self.G.current_node_number):
            if self.G.node_attributes_attachment['adoption'][node_id] == 1:
                adoption_each_step[self.G.node_attributes_attachment['adoption_time'][node_id]] += 1
        temp_sum = 0
        adoption_sum_list = []
        for i in range(self.simulation_time_length):
            temp_sum += adoption_each_step[i]
            adoption_sum_list.append(temp_sum)
        adoption_each_step = np.array(adoption_each_step)
        adoption_sum_list = np.array(adoption_sum_list)
        week_list = [i for i in range(self.simulation_time_length)]
        d_each = {'week': week_list, 'number': (adoption_each_step/self.G.scale).astype(int)}
        d_sum = {'week': week_list, 'number': (adoption_sum_list/self.G.scale).astype(int)}

        each_dataframe = pd.DataFrame(data=d_each)
        sum_dataframe = pd.DataFrame(data=d_sum)
        each_dataframe.to_csv("../Results/weekly_adoption_5_percent.csv" )
        sum_dataframe.to_csv("../Results/sum_adoption_5_percent.csv" )


    def output_heatmap(self):
        week_list = []
        zipcode_list = []
        for node_id in range(self.G.current_node_number):
            if self.G.node_attributes_attachment['adoption'][node_id] == 1:
                week_list.append(self.G.node_attributes_attachment["adoption_time"][node_id])
                zipcode_list.append(self.G.node_attributes_attachment["zipcode"][node_id])
        d = {'week': week_list, 'zipcode': zipcode_list}
        output_dataframe = pd.DataFrame(data=d)
        output_dataframe.to_csv("../Results/heatmap_data.csv" )

    def output_cumulative_sum_by_zipcode(self, simulation_time_length):
        # simulation_time_length = simulation_time_length
        zipcode_dict = {}
        for node_id in range(self.G.current_node_number):
            if self.G.node_attributes_attachment['zipcode'][node_id] not in zipcode_dict:
                zipcode_dict[self.G.node_attributes_attachment['zipcode'][node_id]] = [0] * simulation_time_length
            if self.G.node_attributes_attachment['adoption'][node_id] == 1:
                zipcode_dict[self.G.node_attributes_attachment['zipcode'][node_id]][self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
            # else:
            #     zipcode_dict[self.G.node_attributes_attachment['zipcode'][node_id]][self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
        zipcode_number = len(zipcode_dict.keys())
        dataframe_length = zipcode_number * (simulation_time_length-1)
        array = np.array(np.zeros((dataframe_length, 3)))
        output_dataframe = pd.DataFrame(array, columns=["zipcode","week", "number"])
        temp_index = 0
        for key in zipcode_dict.keys():
            temp_sum = 0
            for i in range(simulation_time_length-1):
                temp_sum += zipcode_dict[key][i]
                output_dataframe.iloc[temp_index] = [key, i, int(temp_sum / self.G.scale)]
                temp_index += 1

        # output_dataframe.to_csv("../Results/output.csv")

        mypath = os.path.join(
            os.path.dirname(__file__), '..', 'Results')
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        onlyfiles = [f for f in onlyfiles if (".csv" in f) and ("out" in f)]
        output_dataframe.to_csv("../Results/output_%s.csv" %(len(onlyfiles)))


if __name__ == "__main__":
    for i in range(1):
        WA_ZIPCODE_COORDINATES_PATH = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'wa_zipcode_coordinates.csv'
        )
        print( WA_ZIPCODE_COORDINATES_PATH)
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
        path = os.path.join(
            os.path.dirname(__file__), '..', 'Data', 'wa_pev_weekly_reg.csv'
        )

        G = NetworkCreatorNetworkit(30000)
        G.generate_node_attribute_attachment(attribute_dict, attribute_type_dict)
        csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'BEV_data.csv'))
        G.generate_nodes_from_population_income_csv(csv_path=csv_path)
        G.generate_edge_list(WA_ZIPCODE_COORDINATES_PATH, M_PATH)
        G.generate_edges()
        G.set_node_degree()

        simulation_paras = {"income_coeff": 9.1e-6,
                            "neighbor_adoption_coeff": 8.62e-3}
        #860 5%
        simulation_time_length = 611
        simulation = Simulation(G, simulation_time_length, simulation_paras)
        simulation.run()
        # simulation.output_weekly_cumulative_adoption()
        #simulation.output_cumulative_sum_by_zipcode(simulation_time_length)
        # simulation.show_adoption_history(path)
        # print("absolute error", simulation.calculate_absolute_error(path))
        # simulation.output_cumulative_sum_by_zipcode(simulation_time_length)
        simulation.output_heatmap()
