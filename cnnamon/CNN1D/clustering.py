import numpy as np
import pandas as pd
import os
import warnings
from tensorflow import keras
import tensorflow as tf
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pycirclize import Circos
from matplotlib.colors import SymLogNorm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", module="absl")


class FilterClustering:


    def __init__(
        self,
        model: keras.Model,
        testset: Dict[str, np.ndarray],
        target_layer: Optional[str] = None,
        linkage_method: str = "ward",
    ):
        print("Initializing FilterClustering...")

        self.model = model
        self.test_x = testset["x"]
        self.target_layer = self._find_target_conv1d(target_layer)
        self.num_filters = self.target_layer.filters
        self.filter_names = [f"Filter{i}" for i in range(self.num_filters)]
        activations = self._get_activations()
        self.fingerprints = self._get_fingerprints(activations)
        self.norm_fingerprints = self._normalize_fingerprints(self.fingerprints)
        print(f"Clustering {self.num_filters} filters using '{linkage_method}' linkage...")
        self.linkage_matrix = sch.linkage(
            self.norm_fingerprints.T, method=linkage_method
        )
        self.newick_tree = self._to_newick(
            self.linkage_matrix, self.filter_names
        )

        self.optimal_k: Optional[int] = None
        self.silhouette_scores: Dict[int, float] = {}
        self.suggest_optimal_clusters(plot=True)

        print("FilterClustering ready.")

    def _find_target_conv1d(self, layer_name):
        if layer_name:
            layer = self.model.get_layer(layer_name)
            if not isinstance(layer, keras.layers.Conv1D):
                raise ValueError("Target layer is not Conv1D")
            return layer

        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv1D):
                return layer

        raise ValueError("No Conv1D layer found.")

    def _get_activations(self):
        activation_model = keras.Model(
            inputs=self.model.input, outputs=self.target_layer.output
        )
        print(f"Calculating activations for {self.test_x.shape[0]} samples...")
        activations = activation_model.predict(
            self.test_x, batch_size=512, verbose=0
        )
        return np.maximum(activations, 0)

    def _get_fingerprints(self, activations):
        print("Creating activation fingerprints...")
        avg_act = np.mean(activations, axis=1)
        return pd.DataFrame(avg_act, columns=self.filter_names)

    def _normalize_fingerprints(self, df):
        print("Applying Z-score normalization...")
        scaler = StandardScaler()
        return pd.DataFrame(
            scaler.fit_transform(df), columns=df.columns
        )

    def _get_newick_string(self, node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parentdist - node.dist:.2f}{newick}"
        else:
            if newick:
                newick = f"):{parentdist - node.dist:.2f}{newick}"
            else:
                newick = ");"
            newick = self._get_newick_string(
                node.get_left(), newick, node.dist, leaf_names
            )
            newick = self._get_newick_string(
                node.get_right(), f",{newick}", node.dist, leaf_names
            )
            return f"({newick}"

    def _to_newick(self, linkage_matrix, leaf_names):
        tree = sch.to_tree(linkage_matrix, False)
        return self._get_newick_string(tree, "", tree.dist, leaf_names)


    def suggest_optimal_clusters(self, max_k=10, plot=True):
        print("Running silhouette analysis...")

        X = self.norm_fingerprints.T
        limit = min(max_k, self.num_filters - 1)

        if limit < 2:
            self.optimal_k = 1
            return 1

        scores = {}
        for k in range(2, limit + 1):
            labels = sch.fcluster(
                self.linkage_matrix, t=k, criterion="maxclust"
            )
            scores[k] = silhouette_score(X, labels)

        self.silhouette_scores = scores
        self.optimal_k = max(scores, key=scores.get)

        print(f"Optimal k = {self.optimal_k}")

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(scores.keys(), scores.values(), marker="o")
            plt.axvline(self.optimal_k, linestyle=":", color="red")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette score")
            plt.title("Silhouette Analysis")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return self.optimal_k

    def plot_circlize(self, n_clusters=None, savefig=None):
        """
        Plots the filter clustering with a dendrogram and outer cluster coloring
        using pycirclize's TreeVisualizer.highlight().
        """
        print("Plotting Circos dendrogram with cluster colors...")

        if n_clusters is None:
            n_clusters = self.optimal_k or 0
        circos, tv = Circos.initialize_from_tree(
            self.newick_tree,
            r_lim=(30, 90),
            leaf_label_size=10 if self.num_filters < 50 else 0,
            line_kws=dict(color="black", lw=1),
        )

        if n_clusters > 1:
            labels = sch.fcluster(self.linkage_matrix, t=n_clusters, criterion="maxclust")
            filter_to_cluster = dict(zip(self.filter_names, labels))

            clusters = {}
            for fname, cid in filter_to_cluster.items():
                clusters.setdefault(cid, []).append(fname)
            palette = sns.color_palette("husl", n_clusters)
            for cid, leaves in clusters.items():
                color = palette[cid - 1]
                tv.highlight(leaves, color=color, alpha=0.5)
            handles = [
                mpatches.Patch(color=palette[i], label=f"Cluster {i + 1}")
                for i in range(n_clusters)
            ]
            fig = circos.plotfig()
            fig.legend(
                handles=handles,
                title="Clusters",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        else:
            fig = circos.plotfig()
        if savefig:
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
            print(f"Circlize plot saved to {savefig}")
        else:
            plt.show()

    def get_clusters(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.optimal_k

        labels = sch.fcluster(
            self.linkage_matrix,
            t=n_clusters,
            criterion="maxclust",
        )

        df = pd.DataFrame(
            {
                "cluster": labels,
            },
            index=self.filter_names,
        )

        df.index.name = "filter_name"
        return df


    def plot_heatmap(self, savefig=None):
        sns.set_theme(style="white")

        g = sns.clustermap(
            self.norm_fingerprints.T,
            row_linkage=self.linkage_matrix,
            cmap="vlag",
            norm=SymLogNorm(linthresh=1),
            dendrogram_ratio=(0.15, 0.05),
            yticklabels=self.num_filters < 40,
        )

        g.fig.suptitle("Filter Activation Heatmap", y=1.02)

        if savefig:
            plt.savefig(savefig, dpi=300)
        else:
            plt.show()
