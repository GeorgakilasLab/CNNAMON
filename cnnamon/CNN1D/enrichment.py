import numpy as np
import pandas as pd
import os
import warnings
import sys
from tensorflow import keras
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, Optional, List, Union, Tuple
from scipy.stats import mannwhitneyu, ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="absl")

def _compute_fold_change(filter_activations: np.ndarray, y_labels: np.ndarray, method='mann-whitney', filter = int(0)):
    num_classes = y_labels.shape[1]
    p_values = []
    logFC_values = []

    for k in range(num_classes):
        pos = filter_activations[y_labels[:, k] == 1]
        neg = filter_activations[y_labels[:, k] == 0]

        mean_pos = np.mean(pos) if len(pos) > 0 else 0.0
        mean_neg = np.mean(neg) if len(neg) > 0 else 0.0
        logFC = np.log2((mean_pos + 1e-6) / (mean_neg + 1e-6))
        logFC_values.append(logFC)

        if len(pos) < 1 or len(neg) < 1:
            p_values.append(1.0)
            continue

        try:
            if method == 'mann-whitney':
                _, p = mannwhitneyu(pos, neg, alternative='two-sided')
            else:
                p = 1.0
            p_values.append(p)
        except ValueError:
            p_values.append(1.0)

    return p_values, logFC_values, filter


def _compute_odds_ratio(motifset, universe: np.ndarray):

    filters = list(motifset.keys())
    df_rows = []
    total_positive = 0
    total_negative = 0
    
    if universe == "activated":
        for f in filters:
            if np.sum(motifset.subseq_classes[f] == 1):
                total_positive+= np.sum(motifset.subseq_classes[f] == 1)
            if np.sum(motifset.subseq_classes[f] == 0):
                total_negative+= np.sum(motifset.subseq_classes[f] == 0)

    elif universe == "global":
        total_positive = np.sum(motifset.labels_subseq_all == 1)
        total_negative = np.sum(motifset.labels_subseq_all == 0)
    else:
        sys.exit("Not a valid universe option. Selelect between activated and global.")

    for f in filters:
        a = np.sum(motifset.subseq_classes[f] == 1)
        c = np.sum(motifset.subseq_classes[f] == 0)
        b = total_positive - a
        d = total_negative - c
        table = np.array([[a, b], [c, d]])
        try:
            odds_ratio, p_value = fisher_exact(table)
        except:
            odds_ratio, p_value = np.nan, 1.0

        log_odds = np.log(odds_ratio + 1e-6)

        df_rows.append({
            'filter': f, 'a': a, 'b': b, 'c': c, 'd': d,
            'odds_ratio': odds_ratio, 'log_odds': log_odds,
            'pvalue': p_value
        })

    df = pd.DataFrame(df_rows)

    df['qvalue'] = multipletests(df['pvalue'], method='fdr_bh')[1]
    return df

class FilterEnrichment:
    def __init__(self,
                 model: keras.Model,
                 data: Dict[str, np.ndarray],
                 motifset: Optional[object] = None,
                 universe: str = "activated",
                 method: str = 'fold_change',
                 n_cores: int = 1):
        self.method = method
        self.model = model
        self.data = data
        self.motifset = motifset
        self.n_cores = n_cores
        self.universe = universe

        if method == 'fold_change':
            self._run_fold_change()
        elif method == 'odds_ratio':
            if motifset is None:
                raise ValueError("motifset must be provided for odds_ratio method")
            self._run_odds_ratio()
        else:
            raise ValueError("method must be 'fold_change' or 'odds_ratio'")

    def _run_fold_change(self):
        layer = self.model.layers[0] if not hasattr(self, 'target_layer') else self.model.get_layer(self.target_layer)
        act_model = keras.Model(self.model.input, layer.output)
        activations = act_model.predict(self.data['x'], batch_size=256, verbose=1)
        fingerprints = np.max(np.where(activations<0, 0, activations), axis=1)
        num_filters = fingerprints.shape[1]
        y_labels = self.data['y']
        results = Parallel(n_jobs=self.n_cores)(delayed(_compute_fold_change)(
            fingerprints[:, i], y_labels,filter = i
        ) for i in tqdm(range(num_filters), desc="FoldChange Enrichment"))
        pvals = np.array([r[0] for r in results])
        logFCs = np.array([r[1] for r in results])
        filters = np.array([r[2] for r in results])
        self.df = pd.DataFrame(logFCs, index=[f"filter{i}" for i in filters],
                               columns=[f"Class_{i}" for i in range(y_labels.shape[1])])
        self.pvals = pd.DataFrame(pvals, index=filters,
                                  columns=[f"Class_{i}" for i in range(y_labels.shape[1])])
        qvals_flat = multipletests(self.pvals.values.flatten(), method='fdr_bh')[1]
        self.qvals = pd.DataFrame(qvals_flat.reshape(self.pvals.shape),
                                  index=self.pvals.index, columns=self.pvals.columns)
        self.plot_data = self.df
        

    def _run_odds_ratio(self):
        df = _compute_odds_ratio(self.motifset, self.universe)
        df = df.sort_values('log_odds', ascending=False)
        df.index = df['filter']
        self.df = df
        self.plot_data = df['log_odds']
        self.plot_data = df['filter']

    def plot_heatmap(self, q_cutoff=0.05, figsize=None, cmap='vlag', center=0, savefig: Optional[str] = None):
        if figsize is None:
            figsize = (8, max(6, len(self.plot_data) * 0.3))
        sns.set_theme(style="white")
        plt.figure(figsize=figsize)
        if self.method == 'fold_change':
            data = self.df.copy()
            annot = data.round(2).astype(str)
            sig_mask = self.qvals <= q_cutoff
            annot = annot.where(~sig_mask, annot + " *")
            sort_order = data.max(axis=1).sort_values(ascending=False).index
            data = data.loc[sort_order]
            annot = annot.loc[sort_order]
            sns.heatmap(data, annot=annot, fmt="s", linewidths=0.5, cmap=cmap, center=center)
        else:
            data = self.df[['log_odds']].copy()
            annot = data.applymap(lambda x: f"{x:.2f}")
            annot[self.df['qvalue'] <= q_cutoff] += " *"
            sns.heatmap(data, annot=annot, fmt="s", linewidths=0.5, cmap=cmap, center=center)

        plt.ylabel("Filters")
        plt.xlabel("Classes" if self.method=='fold_change' else "Log Odds")
        plt.title(f"Filter Enrichment ({self.method})")
        plt.tight_layout()
        if savefig:
            os.makedirs(os.path.dirname(savefig) or '.', exist_ok=True)
            plt.savefig(savefig, dpi=150)
            print(f"Sensitivity plot saved to {savefig}")
        plt.show()

