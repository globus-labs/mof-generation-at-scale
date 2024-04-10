import typing as t
import networkx as nx
import numpy as np

from collections.abc import Collection
from collections import defaultdict
from examol.utils.conversions import convert_string_to_nx
from scipy import stats
from sklearn.cluster import KMeans

from mofa.model import LigandDescription

T = t.TypeVar("T")


class AbsoluteScorer(t.Protocol):
    """A protocol for scoring a single item."""

    def __call__(self, item: T) -> float:
        pass


class RelativeScorer(t.Protocol):
    """A protocol for scoring a collection of items."""

    def __call__(self, items: Collection[T]) -> list[float]:
        pass


class GraphFeatureScorer:

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def __call__(self, graphs: Collection[nx.Graph]) -> list[float]:
        # Collect the features for each graph, compute the means of each feature (across graphs),
        # and then calculate the delta of each graph's feature score from the mean. These deltas
        # are used
        graph_feature_scores = self._score_graph_features(graphs)
        feature_means = graph_feature_scores.mean(axis=0)
        feature_deltas = np.array([
            np.abs(score - feature_means)
            for score in graph_feature_scores
        ])

        # Use KMeans to cluster all the graphs based on their features. This essentially
        # acts as our measurement of the metric space which we will use for diverse sampling.
        try:
            clustering = KMeans(n_clusters=self.n_clusters).fit(feature_deltas)
        except ValueError:
            clustering = KMeans(n_clusters=len(graphs)).fit(feature_deltas)

        clusters = defaultdict(list)
        for g, label in enumerate(clustering.labels_):
            clusters[label].append(g)

        # We set the priorities of the graphs in a per-cluster basis. The "value" of a graph
        # from each cluster initially starts at 0. As we iterate through the graphs, we decrement
        # the value of their respective cluster accordingly. This is to reduce the priority of future
        # graphs in the same cluster in hopes of ensuring some sense of diversity.
        cluster_priorities = {c: 0 for c, cluster in clusters.items()}
        graph_priorities = []
        for g in range(len(graphs)):
            c = clustering.labels_[g]
            p = cluster_priorities[c]
            graph_priorities.append(p)

            # Decrement the priority of this cluster since it is now "less valuable"
            # since having a graph sampled from it.
            cluster_priorities[c] -= 1

        return graph_priorities

    def _score_graph_features(self, graphs: Collection[nx.Graph]) -> np.ndarray:
        features = [self._get_features(g) for g in graphs]
        scores = stats.zscore(features)
        scores = stats.nan_to_num(scores)
        return scores

    @staticmethod
    def _get_features(graph: nx.Graph) -> np.ndarray:
        """Collects some simple graphical features from a networkx graph.

        The features we consider for this work are as follows:
        1. number of nodes
        2. number of edges
        3. avg. clustering
        4. diameter
        5. average shortest path length
        6. average neighbor degree

        Args:
            graph (nx.Graph): The graph to extract features from.

        Notes:
            This function *could* be a source of lag if the graphs are large. Based on
            initial observations of the size of the graphs built from the ligands, this
            should not be an issue. This becomes more of a problem when getting to the
            regime of graphs close to 1000 nodes. Our graphs seem to be in the realm of
            a few dozen nodes.

        Returns:
            Array with each of the features.
        """
        features = [
            graph.number_of_nodes(),
            graph.number_of_edges(),
            nx.average_clustering(graph),
            nx.diameter(graph),
            nx.average_shortest_path_length(graph),
            np.mean(list(nx.average_neighbor_degree(graph).values())),
        ]
        return np.array(features)


class LigandGraphFeatureScorer(GraphFeatureScorer):
    """
    A simple wrapper class for `GraphFeatureScorer` that takes `LigandDescription` objects as input.
    It will first pull out the SMILES strings from each `LigandDescription` and create networkx graphs
    for each. It then passes this list of graphs into the `__call__` implementation of `GraphFeatureScorer`.
    """

    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)

    def __call__(self, ligands: Collection[LigandDescription]) -> list[float]:
        graphs = [convert_string_to_nx(lig.smiles) for lig in ligands]
        return super().__call__(graphs)
