# environment_multi.py
import networkx as nx
import random

class KGEnvironmentMulti:
    def __init__(self, triples, entity_type_map):
        self.graph = nx.MultiDiGraph()
        self.entity_type_map = entity_type_map  # {entity_id: "Paper"/"Topic"/"FOS"}
        self.build_graph(triples)
    
    def build_graph(self, triples):
        for head, relation, tail in triples:
            self.graph.add_edge(head, tail, relation=relation)

    def get_possible_actions(self, current_entity):
        """Returns list of (relation, next_entity, next_entity_type) from current_entity"""
        if current_entity not in self.graph:
            return []
        return [
            (data['relation'], v, self.entity_type_map.get(v, "Unknown"))
            for u, v, data in self.graph.out_edges(current_entity, data=True)
        ]

    def step(self, current_entity, chosen_relation):
        """Take an action and move to the next entity"""
        actions = self.get_possible_actions(current_entity)
        next_candidates = [(rel, v) for rel, v, _ in actions if rel == chosen_relation]
        if next_candidates:
            _, next_entity = random.choice(next_candidates)
            return next_entity
        else:
            return None
