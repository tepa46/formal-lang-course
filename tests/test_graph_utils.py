import pytest
import networkx as nx
import cfpq_data
from pathlib import Path
from project.graph_utils import (
    get_graph_basic_info,
    create_and_save_labeled_two_cycles_graph,
    graph_to_nfa,
    GraphBasicInfo,
)

TMP_PATH = Path(Path(__file__).parent, "test_tmp_dir")
EXPECTED_PATH = Path(Path(__file__).parent, "test_data", "test_graph_utils")


def setup_module():
    Path.mkdir(TMP_PATH)


def teardown_module():
    for item in TMP_PATH.iterdir():
        item.unlink()
    Path.rmdir(TMP_PATH)


@pytest.mark.parametrize(
    "graph_name, expected_node_number, expected_edges_number, expected_labels",
    [
        ("bzip", 632, 556, ["d", "a"]),
        (
            "univ",
            179,
            293,
            [
                "type",
                "label",
                "subClassOf",
                "domain",
                "range",
                "first",
                "rest",
                "onProperty",
                "someValuesFrom",
                "intersectionOf",
                "subPropertyOf",
                "inverseOf",
                "comment",
                "versionInfo",
            ],
        ),
    ],
)
def test_get_graph_basic_info(
    graph_name, expected_node_number, expected_edges_number, expected_labels
):
    expected_graph_basic_info = GraphBasicInfo(
        expected_node_number, expected_edges_number, expected_labels
    )
    actual_graph_basic_info = get_graph_basic_info(graph_name)

    assert expected_graph_basic_info == actual_graph_basic_info


@pytest.mark.parametrize(
    "first_cycle_nodes_number, second_cycle_nodes_number, labels, expected_graph_path",
    [
        (5, 2, ("7", "8"), Path(EXPECTED_PATH, "graph_example_1.dot")),
        (
            1,
            5,
            ("8", "1"),
            Path(EXPECTED_PATH, "graph_example_2.dot"),
        ),
    ],
)
def test_create_and_save_labeled_two_cycles_graph(
    first_cycle_nodes_number, second_cycle_nodes_number, labels, expected_graph_path
):
    expected_graph = nx.nx_pydot.read_dot(expected_graph_path)

    tmp_filepath = Path(TMP_PATH, "tmp_graph.dot")
    create_and_save_labeled_two_cycles_graph(
        first_cycle_nodes_number, second_cycle_nodes_number, labels, tmp_filepath
    )
    actual_graph = nx.nx_pydot.read_dot(tmp_filepath)

    assert nx.utils.graphs_equal(expected_graph, actual_graph)


@pytest.mark.parametrize(
    "graph_path, start_states, final_states",
    [
        (Path(EXPECTED_PATH, "graph_example_1.dot"), {4}, {6}),
        (Path(EXPECTED_PATH, "graph_example_2.dot"), {0, 1}, {5, 4}),
    ],
)
def test_graph_to_nfa(graph_path, start_states, final_states):
    graph = nx.nx_pydot.read_dot(graph_path)

    res = graph_to_nfa(graph, start_states, final_states)

    assert res.start_states == start_states
    assert res.final_states == final_states
    assert res.symbols == set(cfpq_data.get_sorted_labels(graph))
