import numpy as np
import pandas as pd
import os
import sys
import math
import warnings
import tempfile
from tensorflow import keras
import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", module="absl")

def _importance_helper_get_perturbed_weights(
        weights_list_copy: List[np.ndarray],
        filter_index: int,
        conv_weights_index: int,
        filter_width: int,
        rng: np.random.Generator
) -> List[np.ndarray]:

    conv_weights = weights_list_copy[conv_weights_index]
    filter_slice = conv_weights[:, :, filter_index]
    flat_slice = filter_slice.flatten()
    rng.shuffle(flat_slice)
    conv_weights[:, :, filter_index] = flat_slice.reshape(filter_slice.shape)

    return weights_list_copy



def _importance_helper_evaluate_loss(
        model: keras.Model,
        test_x: np.ndarray,
        test_y: np.ndarray,
        batch_size: int
) -> float:
    evaluation_result = model.evaluate(
        test_x,
        test_y,
        verbose=0,
        batch_size=batch_size
    )

    if isinstance(evaluation_result, list):
        loss = evaluation_result[0]
    else:
        loss = evaluation_result
    return loss


def _run_perturbation_job(
        filter_index: int,
        n_iterations: int,
        model_config: dict,
        original_model_weights: List[np.ndarray],
        conv_weights_index: int,
        filter_width: int,
        test_x: np.ndarray,
        test_y: np.ndarray,
        batch_size: int
) -> List[float]:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings("ignore", module="absl")
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    process_seed = (os.getpid() * 100) + filter_index
    rng = np.random.default_rng(seed=process_seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if 'layers' in model_config and model_config.get('name', '').startswith('sequential'):
            temp_model = tf.keras.models.Sequential.from_config(model_config)
        else:
            temp_model = tf.keras.models.Model.from_config(model_config)

        temp_model.compile(loss='binary_crossentropy')

    losses = []

    for i in range(n_iterations):
        perturbed_weights = [w.copy() for w in original_model_weights]

        perturbed_weights = _importance_helper_get_perturbed_weights(
            perturbed_weights,
            filter_index,
            conv_weights_index,
            filter_width,
            rng 
        )

        temp_model.set_weights(perturbed_weights)
        loss = _importance_helper_evaluate_loss(
            temp_model,
            test_x,
            test_y,
            batch_size
        )
        losses.append(loss)

    return losses


class FilterImportance:

    def __init__(self,
                 model: keras.Model,
                 testset: Dict[str, np.ndarray],
                 n_iterations: int = 10,
                 method: str = 'mean',
                 n_cores: int = 1,
                 batch_size: int = 32
                 ):

        if method not in ['mean', 'median']:
            raise ValueError("Method must be 'mean' or 'median'.")

        print("Initializing FilterImportance experiment...")
        self.model = model

        print("Using the full test set.")
        self.test_x = testset['x']
        self.test_y = testset['y']

        self.n_iterations = n_iterations
        self.method = method
        self.n_cores = n_cores
        self.batch_size = batch_size

        self.model_config = self.model.get_config()
        self.original_model_weights = self.model.get_weights()
        self.baseline_loss: float = 0.0
        self.perturbation_results: Optional[pd.DataFrame] = None
        self.filter_importance_ranking: List[str] = []
        self.conv_layer_index, self.conv_layer = self._find_first_conv1d()
        self.conv_weights_index = self._find_conv_weight_index()

        weights, biases = self.conv_layer.get_weights()
        self.num_filters = weights.shape[2]
        self.filter_width = weights.shape[0]
        self._run_experiment()
        self._calculate_importance()
        print("FilterImportance experiment complete. Ready to plot.")

    def __del__(self):

        pass


    def _find_first_conv1d(self) -> Tuple[int, keras.layers.Layer]:

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, keras.layers.Conv1D):
                print(
                    f"Found target Conv1D layer: '{layer.name}' at index {i}")
                return i, layer
        raise ValueError("Could not find a Conv1D layer in the model.")

    def _find_conv_weight_index(self) -> int:
        target_name = self.conv_layer.weights[0].name

        for i, weight_tensor in enumerate(self.model.weights):
            if weight_tensor.name == target_name:
                return i

        raise ValueError("Could not find Conv1D weights in model.weights")

    def _run_experiment(self):

        print(f"Calculating baseline loss (batch_size={self.batch_size})...")

        evaluation_result = self.model.evaluate(
            self.test_x,
            self.test_y,
            verbose=1,
            batch_size=self.batch_size
        )
        if isinstance(evaluation_result, list):
            self.baseline_loss = evaluation_result[0]
        else:
            self.baseline_loss = evaluation_result

        print(f"Baseline loss: {self.baseline_loss:.6f}")

        print(f"Running perturbation for {self.num_filters} filters "
              f"({self.n_iterations} iterations each) "
              f"using {self.n_cores} cores...")

        results_list = Parallel(n_jobs=self.n_cores)(
            delayed(_run_perturbation_job)(
                i,
                self.n_iterations,
                self.model_config,
                self.original_model_weights,
                self.conv_weights_index,
                self.filter_width,
                self.test_x,
                self.test_y,
                self.batch_size
            )
            for i in tqdm(range(self.num_filters), desc="Perturbing filters")
        )

        filter_names = [f"filter{i}" for i in range(self.num_filters)]
        self.perturbation_results = pd.DataFrame(
            data=np.transpose(results_list),
            columns=filter_names
        )

    def _calculate_importance(self):
        if self.perturbation_results is None:
            raise RuntimeError("Run experiment first.")

        if self.method == 'mean':
            avg_loss = self.perturbation_results.mean(axis=0)
        else:  
            avg_loss = self.perturbation_results.median(axis=0)

        importance_score = avg_loss - self.baseline_loss

        self.filter_importance_ranking = importance_score.sort_values(
            ascending=False
        ).index.tolist()
        self.importance_score = importance_score


    def _plot(self, plot_type: str, savefig: Optional[str] = None, **kwargs):
        if self.perturbation_results is None:
            raise RuntimeError("No results to plot. Run experiment first.")

        sns.set_theme(style="whitegrid", palette="coolwarm")

        plot_data = self.perturbation_results[self.filter_importance_ranking]
        plot_data_long = pd.melt(
            plot_data,
            var_name='Filter',
            value_name='Loss'
        )

        num_filters = len(self.filter_importance_ranking)
        fig_width = max(10, num_filters * 0.5)
        fig_height = 7

        plt.figure(figsize=kwargs.get('figsize', (fig_width, fig_height)))

        if plot_type == 'box':
            sns.boxplot(
                x='Filter',
                y='Loss',
                data=plot_data_long,
                color="lightsteelblue",
                order=self.filter_importance_ranking,
                legend=False
            )
        elif plot_type == 'violin':
            sns.violinplot(
                x='Filter',
                y='Loss',
                data=plot_data_long,
                inner="quartile",
                palette="coolwarm",
                order=self.filter_importance_ranking,
                hue='Filter',
                legend=False
            )

        plt.axhline(
            y=self.baseline_loss,
            color='red',
            linestyle=':',
            linewidth=1.5,
            label=f'Baseline Loss ({self.baseline_loss:.4f})'
        )

        plt.title('Filter Importance (Loss vs. Filter Perurbation)',
                  fontsize=16, fontweight='bold')
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Filters', fontsize=12)
        plt.xticks(rotation=90)
        plt.legend(loc='center left',
                   bbox_to_anchor=(1.02, 0.5),
                   frameon=False)

        sns.despine(left=True, bottom=True)

        plt.tight_layout()

        if savefig:
            save_dir = os.path.dirname(savefig)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(savefig)
            print(f"Plot saved to {savefig}")
        else:
            plt.show()

    def boxplot(self, savefig: Optional[str] = None, **kwargs):
        self._plot('box', savefig, **kwargs)

    def violin(self, savefig: Optional[str] = None, **kwargs):
        self._plot('violin', savefig, **kwargs)
