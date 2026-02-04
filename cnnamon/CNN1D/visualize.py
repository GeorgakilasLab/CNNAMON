import numpy as np
import pandas as pd
import os
import sys
import math
import warnings
from tensorflow import keras
import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import logomaker as lm
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", module="absl")

def get_reverse_complement(kernel_matrix: np.ndarray) -> np.ndarray:
    mat = kernel_matrix[::-1, :].copy()
    mat = mat[:, [3, 2, 1, 0]]
    return mat


def max_shift_pearson(k1: np.ndarray, k2: np.ndarray) -> float:
    if k1.ndim == 1: k1 = k1[:, np.newaxis]
    if k2.ndim == 1: k2 = k2[:, np.newaxis]

    length = k1.shape[0]
    n_channels = k1.shape[1]
    total_energy_k1 = np.sum(k1 ** 2)
    total_energy_k2 = np.sum(k2 ** 2)
    if total_energy_k1 == 0 or total_energy_k2 == 0:
        return 0.0

    best_score = -1.0
    epsilon = 1e-9
    for shift in range(-(length - 1), length):
        if shift < 0:
            s1 = k1[:length + shift]
            s2 = k2[-shift:]
        else:
            s1 = k1[shift:]
            s2 = k2[:length - shift]
        v1 = s1.flatten()
        v2 = s2.flatten()
        v1_centered = v1 - np.mean(v1)
        v2_centered = v2 - np.mean(v2)
        numerator = np.sum(v1_centered * v2_centered)
        denom = np.sqrt(np.sum(v1_centered**2) * np.sum(v2_centered**2)) + epsilon
        raw_pearson = numerator / denom
        energy_s1 = np.sum(s1 ** 2)
        energy_s2 = np.sum(s2 ** 2)
        fraction_1 = energy_s1 / total_energy_k1
        fraction_2 = energy_s2 / total_energy_k2
        penalty_factor = np.sqrt(fraction_1 * fraction_2)
        final_score = raw_pearson * penalty_factor

        if final_score > best_score:
            best_score = final_score

    return best_score

def plot_similarity_heatmap(data, labels, title, savefig, **kwargs):
    plt.figure(figsize=kwargs.get('figsize', (10, 8)))

    tick_labels = labels if len(labels) < 50 else False
    
    sns.heatmap(data, xticklabels=tick_labels, yticklabels=tick_labels,
                cmap=sns.color_palette("Blues", as_cmap=True), square=True, vmin=0, vmax=1,
                cbar_kws={'label': 'Similarity Score'})
    plt.title(title, fontsize=15)
    
    if savefig:
        os.makedirs(os.path.dirname(savefig) or '.', exist_ok=True)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {savefig}")
    plt.show()

def plot_similarity_clustermap(df, title, savefig, **kwargs):
    g = sns.clustermap(df, 
                       col_cluster=False, 
                       cmap=sns.color_palette("YlOrBr", as_cmap=True), 
                       vmin=0, vmax=1, 
                       cbar_pos=(1.02, 0.2, 0.03, 0.6),
                       cbar_kws={'label': 'Similarity Score'},
                       figsize=kwargs.get('figsize', (12, 10)))
    
    g.fig.suptitle(title, y=1.02, fontsize=15)
    
    if savefig:
        os.makedirs(os.path.dirname(savefig) or '.', exist_ok=True)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {savefig}")
    plt.show()


class MotifSet(dict):

    def __init__(self, *args, 
                 background: Dict[str, float] = None, 
                 subsequences: Dict[str, np.ndarray] = None, 
                 **kwargs):
        super(MotifSet, self).__init__(*args, **kwargs)

        if background is None:
            try:
                first_pfm = next(iter(self.values()))
                nuc_count = first_pfm.shape[1]
                nucleotides = list(first_pfm.columns)
                background = {n: 1.0/nuc_count for n in nucleotides}
            except (StopIteration, AttributeError):
                background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

        self.background = pd.Series(background)
        self.redundancy_matrix = None
        self.rc_redundancy_matrix = None
        self.subsequences = subsequences if subsequences is not None else {}

        print(f"MotifSet created with background:\n{self.background}")
        if self.subsequences:
            print(f"Stored activating subsequences for {len(self.subsequences)} filters.")

    @classmethod
    def from_dict(cls, data_dict: Dict, background: Dict[str, float] = None, subsequences=None):
        if background is None and isinstance(data_dict, MotifSet):
            background = data_dict.background.to_dict()
            if subsequences is None:
                subsequences = data_dict.subsequences
        return cls(data_dict, background=background, subsequences=subsequences)


    def filter_redundancy(self, savefig: Optional[str] = None, **kwargs):
        if not self: return
        names = list(self.keys())
        n = len(names)
        matrices = [df.values for df in self.values()]

        arr = np.zeros((n, n))
        print(f"Calculating PFM Redundancy for {n} motifs...")

        for i in tqdm(range(n), desc="PFM Similarity"):
            for j in range(i, n):
                score = max_shift_pearson(matrices[i], matrices[j])
                arr[i, j] = score
                arr[j, i] = score

        self.redundancy_matrix = pd.DataFrame(arr, index=names, columns=names)

        plot_similarity_heatmap(self.redundancy_matrix, names, 
                                "Filter similarity (PFMs)", savefig, **kwargs)

    def filter_RC_redundancy(self, savefig: Optional[str] = None, **kwargs):
        if not self: return
        names = list(self.keys())
        rc_names = [f"RC_{n}" for n in names]
        n = len(names)
        matrices = [df.values for df in self.values()]

        arr = np.zeros((n, n))
        print(f"Calculating PFM RC Redundancy for {n} motifs...")

        for i in tqdm(range(n), desc="PFM RC Similarity"):
            for j in range(n):
                rc_j = get_reverse_complement(matrices[j])
                score = max_shift_pearson(matrices[i], rc_j)
                arr[i, j] = score

        self.rc_redundancy_matrix = pd.DataFrame(arr, index=names, columns=rc_names)

        plot_similarity_clustermap(self.rc_redundancy_matrix, 
                                   "Filter similarity to Reverse complement motif (PFMs)", 
                                   savefig, **kwargs)


    def to_motifs(self, savefig: Optional[str] = None, **kwargs):
        if not self: return
        custom_pal = {"A": "#d62728", "C": "#1f77b4", "G": "#ff7f0e", "T": "#2ca02c"}

        first_pfm = next(iter(self.values()))
        filter_width = first_pfm.shape[0]
        num_filters = len(self)

        grid_cols = 8 if num_filters <= 64 else max(8, math.ceil(num_filters / 10))
        grid_rows = math.ceil(num_filters / grid_cols)
        
        fig_w = grid_cols * ((filter_width * 0.4) + 1.0)
        fig_h = grid_rows * 3.0
        if 'figsize' not in kwargs: kwargs['figsize'] = (fig_w, fig_h)
        sns.set_theme(style="white", palette="bright")
        
        fig, axes = plt.subplots(grid_rows, grid_cols, **kwargs)
        axes_flat = np.array(axes).ravel()
        filter_items = list(self.items())

        for i in range(grid_rows * grid_cols):
            ax = axes_flat[i]
            if i < num_filters:
                name, pfm = filter_items[i]
                if pfm.sum().sum() == 0:
                    ax.text(0.5, 0.5, "No Activation", ha='center', va='center')
                    ax.set_axis_off(); continue

                ic = lm.transform_matrix(pfm + 1e-6, from_type='probability', 
                                        to_type='information', background=self.background.values)
                

                lm.Logo(ic, ax=ax, color_scheme=custom_pal).style_spines(visible=False)
                
                ax.set_ylim(0, 2.0); ax.set_title(name, fontweight='bold')
                if i % grid_cols == 0: ax.set_ylabel('bits')
                else: ax.set_ylabel('')
                ax.set_xticks([])
            else: ax.set_visible(False)
        
        plt.tight_layout()
        if savefig:
            os.makedirs(os.path.dirname(savefig) or '.', exist_ok=True)
            plt.savefig(savefig)
        plt.show()

    def to_meme(self, outfile: str):
        if not self: return
        nucs = list(next(iter(self.values())).columns)
        os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
        with open(outfile, 'w') as f:
            f.write(f"MEME version 4\n\nALPHABET= {' '.join(nucs)}\n\nstrands: + -\n\n")
            f.write(f"Background letter frequencies\n{' '.join([f'{n} {self.background[n]:.4f}' for n in nucs])}\n\n")
            for name, pfm in self.items():
                f.write(f"MOTIF {name}\nletter-probability matrix: alength= {len(nucs)} w= {pfm.shape[0]} nsites= 0 E= 0.0\n")
                f.write(pfm.div(pfm.sum(axis=1)+1e-7, axis=0).to_string(header=False, index=False) + "\n\n")

    def to_svgs(self, outdir: str):
        if not self: return
        os.makedirs(outdir, exist_ok=True)
        for name, pfm in tqdm(self.items(), desc="Exporting SVGs"):
            if pfm.sum().sum() == 0: continue
            ic = lm.transform_matrix(pfm, 'probability', 'information', background=self.background.values)
            plt.figure(figsize=(max(1, pfm.shape[0]*0.4), 3))
            lm.Logo(ic).style_spines(visible=False)
            plt.title(name); plt.ylabel('bits'); plt.ylim(0, 2)
            plt.savefig(os.path.join(outdir, f"{name}.svg"), format="svg", bbox_inches='tight')
            plt.close()

class FilterVisualize:
    NUCLEOTIDES = ["A", "C", "G", "T"]

    @staticmethod
    def filter_redundancy(model: keras.Model, 
                          plot: bool = True,
                          savefig: Optional[str] = None, 
                          **kwargs) -> pd.DataFrame:
        (_, weights, _, num, _, _) = FilterVisualize._get_model_filter_info(model)
        labels = [f"F{i}" for i in range(num)]
        
        arr = np.zeros((num, num))
        print(f"Calculating Raw Kernel Redundancy for {num} filters...")
        
        for i in tqdm(range(num), desc="Kernel Similarity"):
            k1 = weights[:, :, i]
            for j in range(i, num):
                k2 = weights[:, :, j]
                score = max_shift_pearson(k1, k2)
                arr[i, j] = score
                arr[j, i] = score

        df = pd.DataFrame(arr, index=labels, columns=labels)

        if plot:
            plot_similarity_heatmap(df, labels, "Filter similarity (Raw Kernels)", savefig, **kwargs)
        
        return df

    @staticmethod
    def filter_RC_redundancy(model: keras.Model, 
                             plot: bool = True,
                             savefig: Optional[str] = None, 
                             **kwargs) -> pd.DataFrame:
        (_, weights, _, num, _, _) = FilterVisualize._get_model_filter_info(model)
        labels = [f"F{i}" for i in range(num)]
        rc_labels = [f"RC_F{i}" for i in range(num)]
        
        arr = np.zeros((num, num))
        print(f"Calculating Raw Kernel RC Redundancy for {num} filters...")
        
        for i in tqdm(range(num), desc="Kernel RC Similarity"):
            k1 = weights[:, :, i]
            for j in range(num):
                k2_rc = get_reverse_complement(weights[:, :, j])
                arr[i, j] = max_shift_pearson(k1, k2_rc)

        df = pd.DataFrame(arr, index=labels, columns=rc_labels)

        if plot:
            plot_similarity_clustermap(df, "Filter similarity to Reverse complement motif (Raw Kernels)", savefig, **kwargs)
        
        return df

    @staticmethod
    def _get_model_filter_info(model: keras.Model) -> Tuple:
        first_conv = next((l for l in model.layers if isinstance(l, keras.layers.Conv1D)), None)
        if first_conv is None: raise ValueError("No Conv1D layer found.")
        weights = first_conv.get_weights()
        return (first_conv, weights[0], weights[1], weights[0].shape[2], weights[0].shape[0], FilterVisualize.NUCLEOTIDES)

    @staticmethod
    def _compute_background(x_data, nucleotides):
        total = np.sum(x_data, axis=(0, 1))
        s = np.sum(total)
        return {n: c/s if s>0 else 0.25 for n, c in zip(nucleotides, total)}

    @staticmethod
    def _pfm_list_to_motifset(pfm_list, background, subseq_list=None, filter_list=None):
        if filter_list:
            data={f"filter{name}": pfm_list[i] for i, name in enumerate(filter_list)}
        else:
            data = {f"filter{i}": pfm for i, pfm in enumerate(pfm_list)}
        subsequences = None
        if subseq_list is not None:
            subsequences = {f"filter{name}": subseq_list[i] for i, name in enumerate(filter_list)}      
        return MotifSet(data, background=background, subsequences=subsequences)

    @staticmethod
    def softmax(model: keras.Model, background=None) -> MotifSet:
        (_, weights, _, num, _, nucs) = FilterVisualize._get_model_filter_info(model)
        if background is None: background = {n: 0.25 for n in nucs}
        pfms = []
        for i in range(num):
            w = weights[:, :, i]
            e_x = np.exp(w - np.max(w, axis=1, keepdims=True))
            pfms.append(pd.DataFrame(e_x / e_x.sum(axis=1, keepdims=True), columns=nucs))
        return FilterVisualize._pfm_list_to_motifset(pfms, background, subseq_list=None)

    @staticmethod
    def top_activating(model, data, percentile=90.0, include_all_positive=False, n_cores=1):
        x_data = data["x"]
        y_labels = data["y"]
        seq_info = data.get("info", [None] * x_data.shape[0])
        (layer, weights, biases, num_filters, width, nucs) = FilterVisualize._get_model_filter_info(model)
        bg = FilterVisualize._compute_background(x_data, nucs)
        act_model = keras.Model(model.input, layer.output)
        activations = act_model.predict(x_data, batch_size=256, verbose=1)
        n, input_len, _ = x_data.shape
        padding_type = getattr(layer, "padding", "valid")
        valid_length = input_len - width + 1
        input_len = x_data.shape[1]
        act_len = activations.shape[1]
        valid_length = input_len - width + 1

        if padding_type == "valid":
            pad_left = 0
            pad_right = 0

            trim_start = 0
            trim_end = act_len
            offset = 0

        elif padding_type == "causal":
            pad_left = width - 1
            pad_right = 0

            trim_start = pad_left
            trim_end = pad_left + valid_length
            offset = -pad_left

        elif padding_type == "same":
            total_pad = width - 1
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left

            trim_start = pad_left
            trim_end = act_len - pad_right
            offset = -pad_left

        else:
            raise ValueError(f"Unsupported padding type: {padding_type}")
        assert trim_start >= 0
        assert trim_end <= act_len
        assert trim_end > trim_start
        activations = activations[:, trim_start:trim_end, :]
        per_seq = activations.shape[1]
        print(padding_type)
        per_seq = activations.shape[1]
        if per_seq <= 0:
            return MotifSet({}, background=bg)
        labels_subseq_all = np.zeros(n * per_seq, dtype=int)
        idx = 0
        for i in range(n):
            for j in range(per_seq):
                labels_subseq_all[idx] = y_labels[i]
                idx += 1

        def process(i):
            f_act = activations[:, :,i]
            pos_act = f_act[f_act > 0]
            if pos_act.size == 0:
                return (pd.DataFrame(np.zeros((width, 4)), columns=nucs), 
                        None, None, None, None)
            thresh = 0.0 if include_all_positive else np.percentile(pos_act, percentile)
            rows, cols = np.where(f_act >= thresh)
            print(thresh)

            subs, labels, infos = [], [], []
            for r, c in zip(rows, cols):
                subseq = x_data[r, c : c + width]
                subs.append(subseq)
                labels.append(y_labels[r])
                infos.append(",".join(seq_info[r].split(",")[:3]) + f",{ '+' if len(seq_info[r].split(','))==3 else '-' },{c},{c+width}")
            if not subs:
                return None
            else:
                stacked_subs = np.stack(subs)
                pcm = np.sum(stacked_subs, axis=0)
                pfm = pd.DataFrame(pcm / (np.sum(pcm, axis=1, keepdims=True)), columns=nucs)
                return pfm, stacked_subs, np.array(labels), np.array(infos), i

        results = Parallel(n_jobs=n_cores)(
            delayed(process)(i) for i in tqdm(range(num_filters), desc="Processing filters")
        )

        results = [res for res in results if res is not None]
        if not results:
            return MotifSet({}, background=bg)

        pfms, subseq_list, class_list, info_list, filter_list = zip(*results)
        print(filter_list)
        motifset = FilterVisualize._pfm_list_to_motifset(pfms, bg, subseq_list=list(subseq_list), filter_list=filter_list)
        motifset.labels_subseq_all = labels_subseq_all
        motifset.subseq_classes = {f"filter{i}": class_list[idx] for idx, i in enumerate(filter_list)}
        motifset.subseq_info = {f"filter{i}": info_list[idx] for idx, i in enumerate(filter_list)}
        return motifset

    @staticmethod
    def pos_activating(model, n_cores=1, background=None) -> MotifSet:
        (_, weights, _, num, _, nucs) = FilterVisualize._get_model_filter_info(model)
        if background is None: background = {n: 0.25 for n in nucs}
        def process(i):
            mask = (weights[:, :, i] > 0).astype(float)
            sums = np.sum(mask, axis=1, keepdims=True)
            return pd.DataFrame(np.divide(mask, sums, out=np.full_like(mask, 0.25), where=sums!=0), columns=nucs)
        
        pfms = Parallel(n_jobs=n_cores)(delayed(process)(i) for i in tqdm(range(num), desc="Processing"))
        return FilterVisualize._pfm_list_to_motifset(pfms, background, subseq_list=None)
    
    @staticmethod
    def significant_activating(
        model,
        data: Dict[str, np.ndarray],
        n_perturbations=1000,
        q_value_cutoff=1.0,
        n_cores=1
    ) -> "MotifSet":

        x_data = data["x"]
        y_labels = data["y"]
        seq_info = data.get("info", [None] * x_data.shape[0])
        (_, weights, biases, num_filters, width, nucs) = FilterVisualize._get_model_filter_info(model)
        bg = FilterVisualize._compute_background(x_data, nucs)
        n, l, _ = x_data.shape
        per_seq = l - width + 1
        if per_seq <= 0:
            return MotifSet({}, background=bg)
        
        all_subs = np.zeros((n * per_seq, width, 4), dtype=np.float32)
        seq_ids = np.zeros(n * per_seq, dtype=int)
        subseq_starts = np.zeros(n * per_seq, dtype=int)
        labels_subseq_all = np.zeros(n * per_seq, dtype=int)
        idx = 0
        for i in range(n):
            for j in range(per_seq):
                all_subs[idx] = x_data[i, j:j+width, :]
                seq_ids[idx] = i
                subseq_starts[idx] = j
                labels_subseq_all[idx] = y_labels[i]
                idx += 1

        def process(filter_idx):
            w, b = weights[:, :, filter_idx], biases[filter_idx]
            real_act = np.einsum('nwl,wl->n', all_subs, w) + b
            pos_mask = real_act > 0
            pos_subs = all_subs[pos_mask]
            pos_seq_ids = seq_ids[pos_mask]
            pos_starts = subseq_starts[pos_mask]
            pos_real = real_act[pos_mask]

            if len(pos_subs) == 0:
                return None

            exceeded = np.zeros(len(pos_subs))
            for _ in range(n_perturbations):
                noise = np.random.permutation(w.flatten()).reshape(w.shape)
                exceeded += (np.einsum('nwl,wl->n', pos_subs, noise) + b > pos_real)

            pvals = (exceeded + 1) / (n_perturbations + 1)
            _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
            keep = qvals <= q_value_cutoff if q_value_cutoff < 1.0 else np.ones_like(qvals, dtype=bool)

            kept_subs = pos_subs[keep]
            kept_seq_ids = pos_seq_ids[keep]
            kept_starts = pos_starts[keep]
            kept_pvals = pvals[keep]
            kept_qvals = qvals[keep]

            if len(kept_subs) > 0:
                pcm = np.sum(kept_subs, axis=0)
                pfm = pd.DataFrame(pcm / (np.sum(pcm, axis=1, keepdims=True)), columns=nucs)
                class_labels = np.array([y_labels[sid] for sid in kept_seq_ids])

                genomic_info = []

                for sid, sub_start, pval, qval in zip(
                        kept_seq_ids, kept_starts, kept_pvals, kept_qvals):
                    
                    fields = seq_info[sid].split(",")
                    if len(fields) < 3:
                        raise ValueError(f"seq_info[{sid}] must have at least 3 comma-separated fields")
                    
                    chrom, seq_start, seq_end = fields[:3]
                    seq_start = int(seq_start)
                    strand = "+" if len(fields) == 3 else "-"

                    sub_genomic_start = seq_start + sub_start
                    sub_genomic_end   = sub_genomic_start + width

                    info_str = f"{chrom},{seq_start},{seq_end},{strand},{sub_genomic_start},{sub_genomic_end},{pval},{qval}"
                    genomic_info.append(info_str)

                genomic_info = np.array(genomic_info, dtype=str)


                return pfm, kept_subs, class_labels, genomic_info, kept_pvals, kept_qvals, filter_idx
            else:
                return None
        results = Parallel(n_jobs=n_cores)(
            delayed(process)(i) for i in tqdm(range(num_filters), desc="Perturbing filters")
        )
        results = [r for r in results if r is not None]

        if len(results) == 0:
            return MotifSet({}, background=bg)

        pfms, subseq_list, class_list, info_list, pvals_list, qvals_list, filter_list = zip(*results)

        motifset = FilterVisualize._pfm_list_to_motifset(
            pfms, bg, subseq_list=list(subseq_list), filter_list=filter_list
        )
        motifset.labels_subseq_all = labels_subseq_all
        motifset.subseq_classes = {f"filter{filter}": class_list[i] for i, filter in enumerate(filter_list)}
        motifset.subseq_info = {f"filter{filter}": info_list[i] for i, filter in enumerate(filter_list)}
        motifset.q_values = {f"filter{filter}": qvals_list[i] for i, filter in enumerate(filter_list)}
        motifset.p_values = {f"filter{filter}": pvals_list[i] for i, filter in enumerate(filter_list)}

        return motifset


def analyze_nucleotide_sensitivity(model: keras.Model, 
                                   motifset: MotifSet, 
                                   n_cores: int = 1) -> pd.DataFrame:
    if not motifset.subsequences:
        raise ValueError("The provided MotifSet does not contain subsequences. "
                         "Run top_activating or significant_activating first.")
    (layer, weights, biases, num_filters, filter_width, nucs) = FilterVisualize._get_model_filter_info(model)
    valid_mutations = {i: [x for x in range(4) if x != i] for i in range(4)}

    def process_filter(filter_name):
        filter_idx = int(filter_name.replace("filter", "").replace("F", "")) 
        subseqs = motifset.subsequences[filter_name]
        w = weights[:, :, filter_idx]
        b = biases[filter_idx]
        results = []
        
        orig_acts = np.einsum('nwl,wl->n', subseqs, w) + b
        pos_acts = orig_acts[orig_acts > 0]
        print(len(pos_acts
                  ))
        eps = np.percentile(pos_acts, 1)


        for seq_idx, (seq, orig_act) in enumerate(zip(subseqs, orig_acts)):

            if orig_act > eps:  
                seq_indices = np.argmax(seq, axis=1)

                for pos in range(filter_width):
                    original_base_idx = seq_indices[pos]
                    
                    for mut_idx in valid_mutations[original_base_idx]:
                        mut_seq = seq.copy()
                        mut_seq[pos, original_base_idx] = 0
                        mut_seq[pos, mut_idx] = 1
                        mut_act = np.einsum('wl,wl->', mut_seq, w) + b
                        mut_act = max(0, mut_act)  
                        fc = np.log2((mut_act+eps) / (orig_act+eps))



                        
                        results.append({
                            'Filter': filter_name,
                            'SeqID': seq_idx,
                            'Position': pos,
                            'Base': nucs[mut_idx],
                            'FoldChange': fc
                        })
                    
                    results.append({
                        'Filter': filter_name,
                        'SeqID': seq_idx,
                        'Position': pos,
                        'Base': nucs[original_base_idx],
                        'FoldChange': 1.0
                    })

        return results
    all_results = Parallel(n_jobs=n_cores,backend='threading')(
        delayed(process_filter)(fname) for fname in tqdm(motifset.keys(), desc="Mutagenesis")
    )

    flat_results = [item for sublist in all_results for item in sublist]
    return pd.DataFrame(flat_results)

def plot_nucleotide_sensitivity(df: pd.DataFrame, 
                                savefig: Optional[str] = None):
    filters = df['Filter'].unique()
    num_filters = len(filters)
    
    cols = 4
    rows = math.ceil(num_filters / cols)
    figsize = (cols * 5, rows * 4)

    custom_pal = {"A": "#d62728", "C": "#1f77b4", "G": "#ff7f0e", "T": "#2ca02c"}

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
    axes = np.array(axes).ravel()

    for i, f_name in enumerate(filters):
        ax = axes[i]
        subset = df[df['Filter'] == f_name]
        

        sns.boxplot(data=subset, x='Position', y='FoldChange', hue='Base', 
                    palette=custom_pal, ax=ax, fliersize=1, linewidth=0.8,
                    hue_order=["A", "C", "G", "T"])
        
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f_name)
        ax.set_xlabel("Filter Position")
        if i % cols == 0:
            ax.set_ylabel("Log2 Activation Fold Change")
        else:
            ax.set_ylabel("")
            
        if i == 0:
            sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1), ncol=4, frameon=False, title=None)
        else:
            if ax.get_legend(): ax.get_legend().remove()

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if savefig:
        os.makedirs(os.path.dirname(savefig) or '.', exist_ok=True)
        plt.savefig(savefig, dpi=150)
        print(f"Sensitivity plot saved to {savefig}")
    plt.show()

def filter_q_motifset(motifset: MotifSet, q_cutoff: float) -> MotifSet:
    new_pfms = {}
    new_subs = {}
    new_classes = {}
    new_info = {}
    labels_subseq_all = motifset.labels_subseq_all

    for fname, subs in motifset.subsequences.items():
        qvals = motifset.q_values[fname]
        mask = qvals <= q_cutoff
        if not np.any(mask):
            continue
        kept_subs = subs[mask]
        pcm = np.sum(kept_subs, axis=0)
        pfm = pd.DataFrame(
            pcm / (np.sum(pcm, axis=1, keepdims=True) + 1e-7),
            columns=["A", "C", "G", "T"]
        )
        new_pfms[fname] = pfm
        new_subs[fname] = kept_subs
        new_classes[fname] = motifset.subseq_classes[fname][mask]
        new_info[fname] = motifset.subseq_info[fname][mask]

    filtered_motifset = MotifSet(new_pfms, background=motifset.background, subsequences=new_subs)
    filtered_motifset.subseq_classes = new_classes
    filtered_motifset.subseq_info = new_info
    filtered_motifset.labels_subseq_all = labels_subseq_all
    return filtered_motifset

def filter_motifset(motifset: MotifSet, filters: list) -> MotifSet:
    new_pfms = {}
    new_subs = {}
    new_classes = {}
    new_info = {}
    labels_subseq_all = motifset.labels_subseq_all
    for fname, subs in motifset.subsequences.items():
        if fname in filters:
            new_pfms[fname] = motifset[fname]
            new_subs[fname] = subs
            new_classes[fname] = motifset.subseq_classes[fname]
            new_info[fname] = motifset.subseq_info[fname]

    filtered_motifset = MotifSet(new_pfms, background=motifset.background, subsequences=new_subs)
    filtered_motifset.subseq_classes = new_classes
    filtered_motifset.subseq_info = new_info
    filtered_motifset.labels_subseq_all = labels_subseq_all
    return filtered_motifset