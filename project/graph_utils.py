from dataclasses import dataclass
from typing import Tuple, List, Any
import networkx as nx
import cfpq_data


@dataclass
class GraphBasicInfo:
    nodes_number: int
    edges_number: int
    labels: List[Any]


def get_graph_basic_info(graph_name: str) -> GraphBasicInfo:
    graph_csv = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_csv)

    nodes_number = graph.number_of_nodes()
    edges_number = graph.number_of_edges()
    labels = cfpq_data.get_sorted_labels(graph)

    return GraphBasicInfo(
        nodes_number,
        edges_number,
        labels,
    )


def create_and_save_labeled_two_cycles_graph(
    first_cycle_nodes_number: int,
    second_cycle_nodes_number: int,
    labels: Tuple[str, str],
    file_path: str,
):
    graph = cfpq_data.labeled_two_cycles_graph(
        n=first_cycle_nodes_number, m=second_cycle_nodes_number, labels=labels
    )

    graph_in_dot = nx.nx_pydot.to_pydot(graph)
    graph_in_dot.write(file_path)
