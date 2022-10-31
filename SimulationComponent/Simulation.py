import networkx as nx


class Simulation:
    def __init__(self, G, state_transition: callable, simulation_time: int):
        self.state_transition = state_transition
        self.G = G
        self.simulation_time = simulation_time
        self.current_simulation_time = 0

    def step(self):
        #nx.set_node_attributes(self.G, state, 'state')
        pass