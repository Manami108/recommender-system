import networkx as nx
import random

class KGEnvironment:
    def __init__(self, triples):
        self.graph = nx.MultiDiGraph()
        self.build_graph(triples)
    
    def build_graph(self, triples):
        for head, relation, tail in triples:
            self.graph.add_edge(head, tail, relation=relation)

    def get_possible_actions(self, current_entity):
        """Returns list of (relation, next_entity) from current_entity"""
        if current_entity not in self.graph:
            return []
        return [(data['relation'], v) for u, v, data in self.graph.out_edges(current_entity, data=True)]

    def step(self, current_entity, action_relation):
        """Take action (relation), return next entity if exists, else None"""
        actions = self.get_possible_actions(current_entity)
        next_entities = [v for r, v in actions if r == action_relation]
        if next_entities:
            return random.choice(next_entities)
        else:
            return None
