from __future__ import annotations

import os
import re
import subprocess
import tempfile
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore


class PrepareData:
    """
    Gerenares tensor arrays for training, validation and test splits.

    Parameters
    ----------
    intervalfile : str
        File path indicating an interval file in BED4(3+1) file format. 
        First three columns represent interval coordinates (chromosome,
        start,end] and the fourth the label (0/1 for binary classification,
        0/1|0/1|0/1.. for multi-class classification)
    genomefasta : str
        File path indicating a genome file in FASTA format (.fasta, .fa).
    outdir : str
        A directory path where data will be stored. Could be existing or not.
        Based on split_segmentation a nested directory is generated.
    split_segmentation : str
        "random", "chromosome" or "custom".
        random - splits will be generated based on ratios on the full interval file.
        chromosome - splits will be generated based on chromosome ratios (chromosome hold-out)
        custom - splits will be generated based on user defined chr lists (train_chr_list,val_chr_list,test_chr_list)
    ratios : list(float or int) [training ratio, validation ratio, test ratio]
        List specifying the ratios in Train, Validation, Test segmentation. (split_segmentation = "chromosome" or "random")
        List is normalised in case the proportions does not sum to 1.
        e.g. [0.6, 0.2, 0.2] | [6, 2, 2] | [3, 1, 1]
    augment_RC : int - FLAG
        If set to 1, the reverse complement for each sequence is generated for data augmentation.
        Same label with the original interval is set.
    save_splits : int - FLAG
        If set to 1, the dataset will be saved in the specified path (outdir).
    seed : int
        Seed for numpy.random selection in intervals or chromosomes.
    
    If split_segmentation = "custom"
    train_chr_list : list(str)
        List of chromosomes, found in the interval file and designated for training set.
    val_chr_list : list(str)
        List of chromosomes, found in the interval file and designated for validation set.
    test_chr_list : list(str)
        List of chromosomes, found in the interval file and designated for test set.

    Returns
    -------
    DATASET : dictionary
        dict{"x": array of one hot encoded sequences,
             "y": array of labels,
             "info": list of sequence coordinates}
    """

    def __init__(self,
                 intervalfile: str,
                 genomefasta: str,
                 outdir: str,
                 **kwargs):

        # Paths
        self.intervalfile = intervalfile
        self.genomefasta = genomefasta
        self.outdir = outdir
        self.tempdir = kwargs.get("tempdir", tempfile.gettempdir())
        
        # Options
        self.seed = kwargs.get("seed", None)
        self.augment_RC = bool(kwargs.get("augment_RC", False))
        
        # Split config
        # split_segmentation: "random" (intervals), 
        #                     "chromosome" (chr holdout), 
        #                  or "custom" (user defined lists)
        self.split_segmentation = str(kwargs.get("split_segmentation", "random"))
        p = list(kwargs.get("ratios", [0.6, 0.2, 0.2]))
        self.ratios = [x / sum(p) for x in p]
        self.testperc = float(self.ratios[1])
        self.valperc = float(self.ratios[2])
        self.train_chr_list = self._parse_list(kwargs.get("train_chr_list", []))
        self.val_chr_list = self._parse_list(kwargs.get("val_chr_list", []))
        self.test_chr_list = self._parse_list(kwargs.get("test_chr_list", []))
        self.save_splits = str(kwargs.get("save_splits", "0"))
        self.specificchr = str(kwargs.get("specificchr", "0"))
        self.chrlist = self._parse_list(kwargs.get("chrlist", []))
        ###
        self.intervals: Dict[str, List[Tuple[str, str]]] = {}
        self.indexes: List[int] = []
        self.indexed_intervals: Dict[int, str] = {}
        self.label_dim: int = 1
        self.sequence_onehot: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.info_map: Dict[str, List[str]] = {}
        self._info_all: List[str] = []

        self.chromosome_list: List[str] = []
        self._seq_length: int = 0
        self._seq_all: Optional[np.ndarray] = None
        self._lab_all: Optional[np.ndarray] = None

        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.tempdir, exist_ok=True)

        if self.seed is not None:
            self._set_seed(int(self.seed))

    @staticmethod
    def _parse_list(val: Union[str, List[str]]) -> List[str]:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            if not val or val == "0":
                return []
            return [x.strip() for x in val.split(",") if x.strip()]
        return []

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def read_params_from_file(path: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^(.+?)=(.+)$", line)
                if m:
                    k, v = m.group(1).strip(), m.group(2).strip()
                    if "," in v and k.endswith("list"):
                        params[k] = [x.strip() for x in v.split(",")]
                    else:
                        params[k] = v
        return params

    @classmethod
    def from_file(cls, path: str) -> PrepareData:
        params_dict = cls.read_params_from_file(path)
        try:
            return cls(**params_dict)
        except TypeError as e:
            print(f"Error: Missing a required parameter in {path}.")
            raise e

    @staticmethod
    def _load_one_split(directory_path: str,
                        split_name: str) -> Dict[str, np.ndarray | List[str]]:
        seq_file = os.path.join(directory_path, f"{split_name}_sequence.npy")
        lab_file = os.path.join(directory_path, f"{split_name}_Labels.npy")
        info_file = os.path.join(directory_path, f"{split_name}_info.csv")
        data: Dict[str, np.ndarray | List[str]] = {}
        try:
            data["x"] = np.load(seq_file)
            data["y"] = np.load(lab_file)
            with open(info_file, "r") as f:
                info_lines = [line.strip() for line in f.readlines()[1:]]
                data["info"] = info_lines
        except FileNotFoundError:
            data["x"] = np.empty((0, 0, 4), dtype=np.float32)
            data["y"] = np.empty((0, 0), dtype=np.float32)
            data["info"] = []
        return data

    @staticmethod
    def load_splits_from_disk(directory_path: str) -> Tuple[Dict, Dict, Dict]:
        train_data = PrepareData._load_one_split(directory_path, "TrainingSet")
        test_data = PrepareData._load_one_split(directory_path, "TestSet")
        val_data = PrepareData._load_one_split(directory_path, "ValidationSet")
        return train_data, test_data, val_data

    def _infer_label_dimension(self, file: Optional[str] = None) -> int:
        if file is None:
            file = self.intervalfile
        with open(file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) > 3:
                    label_str = parts[3]
                    try:
                        if "|" in label_str:
                            return len(label_str.split("|"))
                        else:
                            float(label_str)
                            return 1
                    except ValueError:
                        continue
        return 1

    def load_intervals(self, file: Optional[str] = None) -> Tuple[
        Dict[str, List[List[str]]], List[int], Dict[int, str]]:
        if file is None:
            file = self.intervalfile
        chrmap: Dict[str, List[List[str]]] = {}
        indexed: Dict[int, str] = {}
        indexes: List[int] = []
        idx = 0
        with open(file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], parts[1], parts[2]
                indexed[idx] = f"{chrom},{start},{end}"
                indexes.append(idx)
                chrmap.setdefault(chrom, []).append([start, end])
                idx += 1
        self.intervals = chrmap
        self.indexes = indexes
        self.indexed_intervals = indexed
        self.label_dim = self._infer_label_dimension(file)
        return chrmap, indexes, indexed

    @staticmethod
    def onehot_encoding(sequence: str) -> np.ndarray:
        # A: [1,0,0,0], 
        # C: [0,1,0,0], 
        # G: [0,0,1,0], 
        # T: [0,0,0,1]
        mapping = {
            "A": np.array([1, 0, 0, 0], dtype=np.float32), "a": np.array([1, 0, 0, 0], dtype=np.float32),
            "C": np.array([0, 1, 0, 0], dtype=np.float32), "c": np.array([0, 1, 0, 0], dtype=np.float32),
            "G": np.array([0, 0, 1, 0], dtype=np.float32), "g": np.array([0, 0, 1, 0], dtype=np.float32),
            "T": np.array([0, 0, 0, 1], dtype=np.float32), "t": np.array([0, 0, 0, 1], dtype=np.float32),
            "N": np.array([0, 0, 0, 0], dtype=np.float32), "n": np.array([0, 0, 0, 0], dtype=np.float32),
        }
        vecs: List[np.ndarray] = [mapping.get(ch, mapping["N"]) for ch in sequence]
        return np.vstack(vecs)

    def sequence_to_array(self, chrom: str, list_of_intervals: List[List[str]]) -> np.ndarray:
        temp_bed = os.path.join(self.tempdir, f"temp_{chrom}.bed")
        with open(temp_bed, "w") as fh:
            for interval in list_of_intervals:
                fh.write(f"{chrom}\t{interval[0]}\t{interval[1]}\n")
        out_seq = os.path.join(self.tempdir, f"sequence_{chrom}.bed")
        cmd = ["bedtools", "getfasta", "-fi", self.genomefasta,
               "-bed", temp_bed, "-bedOut", "-fo", out_seq]
        
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        arrays: List[np.ndarray] = []
        with open(out_seq) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 1: 
                    continue
                seq = parts[0]
                arr = self.onehot_encoding(seq)
                arrays.append(arr)
        
        try:
            os.remove(temp_bed)
            os.remove(out_seq)
        except Exception:
            pass
            
        if not arrays:
            return np.array([], dtype=np.float32)
        return np.asarray(arrays, dtype=np.float32)

    def get_labels_for_chr(self, chrom: str, file: Optional[str] = None) -> np.ndarray:
        if file is None:
            file = self.intervalfile
        labels: List[np.ndarray] = []
        with open(file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if parts[0] != chrom:
                    continue
                
                # Parse Label
                arr: Optional[np.ndarray] = None
                if len(parts) > 3:
                    label_str = parts[3]
                    try:
                        if "|" in label_str:
                            vals = [float(val) for val in label_str.split("|")]
                            arr = np.array(vals, dtype=np.float32)
                        else:
                            arr = np.array([float(label_str)], dtype=np.float32)
                    except ValueError:
                        arr = np.zeros((self.label_dim,), dtype=np.float32)
                else:
                    arr = np.zeros((self.label_dim,), dtype=np.float32)

                if arr is not None:
                    if arr.shape[0] != self.label_dim:
                        arr = np.zeros((self.label_dim,), dtype=np.float32)
                    labels.append(arr)
                    
        if not labels:
            return np.empty(shape=(0, self.label_dim), dtype=np.float32)
        return np.asarray(labels, dtype=np.float32)

    def build_all_chromosomes(self) -> None:
        if not self.intervals:
            self.load_intervals()

        if self.specificchr == "1":
            to_process = [c for c in self.chrlist if c in self.intervals]
        else:
            to_process = list(self.intervals.keys())

        self.chromosome_list = []
        print(f"Processing {len(to_process)} chromosomes...")
        
        for chrom in tqdm(to_process):
            if chrom not in self.intervals:
                continue
            seq_arr = self.sequence_to_array(chrom, self.intervals[chrom])
            lbls = self.get_labels_for_chr(chrom)

            if seq_arr.shape[0] != lbls.shape[0]:
                print(f"WARNING: Mismatch for {chrom}. Seq: {seq_arr.shape[0]}, Lbl: {lbls.shape[0]}. Skipping.")
                continue
            current_intervals = self.intervals[chrom]
            chr_infos = [f"{chrom},{x[0]},{x[1]}" for x in current_intervals]

            # RC
            if self.augment_RC and seq_arr.shape[0] > 0:
                seq_arr_rc = np.flip(seq_arr, axis=(1, 2))
                seq_arr = np.concatenate([seq_arr, seq_arr_rc], axis=0)
                lbls = np.concatenate([lbls, lbls], axis=0)
                rc_infos = [f"{x},RC" for x in chr_infos]
                chr_infos = chr_infos + rc_infos

            if (self._seq_length == 0 and seq_arr.ndim == 3 and seq_arr.shape[0] > 0):
                self._seq_length = seq_arr.shape[1]
            self.sequence_onehot[chrom] = seq_arr
            self.labels[chrom] = lbls
            self.info_map[chrom] = chr_infos

            if chrom not in self.chromosome_list:
                self.chromosome_list.append(chrom)
        if self.chromosome_list:
            seqs = [self.sequence_onehot[c] for c in self.chromosome_list]
            self._seq_all = np.vstack(seqs)
            
            labs = [self.labels[c] for c in self.chromosome_list]
            self._lab_all = np.vstack(labs)
            self._info_all = []
            for c in self.chromosome_list:
                self._info_all.extend(self.info_map[c])
        else:
            print("Warning: No chromosomes processed.")

    def _save_data_split(self, data_dict: Dict, name: str, outdir: str):
        if data_dict["x"].shape[0] == 0:
            print(f"No data for '{name}', skipping save.")
            return

        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, f"{name}_sequence.npy"), data_dict["x"])
        np.save(os.path.join(outdir, f"{name}_Labels.npy"), data_dict["y"])

        info_path = os.path.join(outdir, f"{name}_info.csv")
        with open(info_path, "w") as fh:
            fh.write("chr,intervalstart,intervalend\n")
            fh.write("\n".join(data_dict["info"]))

    def _get_data_by_chr(self, list_of_chr: List[str]) -> Dict[str, np.ndarray | List[str]]:
        valid_chr = [c for c in list_of_chr if c in self.sequence_onehot]
        
        seqs = [self.sequence_onehot[c] for c in valid_chr]
        labs = [self.labels[c] for c in valid_chr]
        info_lines: List[str] = []
        for c in valid_chr:
            info_lines.extend(self.info_map[c])
        
        if not seqs:
            stacked_seq = np.empty((0, self._seq_length, 4), dtype=np.float32)
            stacked_labels = np.empty((0, self.label_dim), dtype=np.float32)
        else:
            stacked_seq = np.vstack(seqs)
            stacked_labels = np.vstack(labs)

        return {"x": stacked_seq, "y": stacked_labels, "info": info_lines}

    def _get_data_by_index(self, indexes: List[int]) -> Dict[str, np.ndarray | List[str]]:
        if self._seq_all is None or self._lab_all is None:
            raise RuntimeError("Cannot get data by index, _seq_all is not built.")
        seqs = [self._seq_all[i, :, :] for i in indexes]
        labs = [self._lab_all[i, :] for i in indexes]
        info_lines = [self._info_all[i] for i in indexes]
        
        if not seqs:
             stacked_seq = np.empty((0, self._seq_length, 4), dtype=np.float32)
             stacked_labels = np.empty((0, self.label_dim), dtype=np.float32)
        else:
             stacked_seq = np.asarray(seqs)
             stacked_labels = np.asarray(labs)

        return {"x": stacked_seq, "y": stacked_labels, "info": info_lines}

    def get_splits(self) -> Tuple[Dict, Dict, Dict]:
        empty_x = np.empty((0, self._seq_length, 4), dtype=np.float32)
        empty_y = np.empty((0, self.label_dim), dtype=np.float32)
        train_data = {"x": empty_x, "y": empty_y, "info": []}
        test_data = {"x": empty_x.copy(), "y": empty_y.copy(), "info": []}
        val_data = {"x": empty_x.copy(), "y": empty_y.copy(), "info": []}

        # user defined "custom" lists
        if self.split_segmentation == "custom":
            print("Splitting by custom chromosome lists...")
            train_data = self._get_data_by_chr(self.train_chr_list)
            val_data = self._get_data_by_chr(self.val_chr_list)
            test_data = self._get_data_by_chr(self.test_chr_list)

        # user defined "chromosome" pick per ratio
        elif self.split_segmentation == "chromosome":
            print("Splitting by chromosome percentages...")
            temp_chr = self.chromosome_list.copy()
            random.shuffle(temp_chr)
            n = len(temp_chr)
            n_test = int(self.testperc * n)
            n_val = int(self.valperc * n)
            test_chr = temp_chr[:n_test]
            val_chr = temp_chr[n_test : n_test + n_val]
            train_chr = temp_chr[n_test + n_val :]
            train_data = self._get_data_by_chr(train_chr)
            val_data = self._get_data_by_chr(val_chr)
            test_data = self._get_data_by_chr(test_chr)

        # user defined "random" intervals
        elif self.split_segmentation == "random":
            print("Splitting by random intervals...")
            if self._seq_all is None:
                print("Error: No data loaded.")
                return train_data, test_data, val_data
            total_samples = self._seq_all.shape[0]
            all_idxs = list(range(total_samples))
            random.shuffle(all_idxs)
            n_test = int(self.testperc * total_samples)
            n_val = int(self.valperc * total_samples)
            test_idxs = all_idxs[:n_test]
            val_idxs = all_idxs[n_test : n_test + n_val]
            train_idxs = all_idxs[n_test + n_val :]
            train_data = self._get_data_by_index(train_idxs)
            val_data = self._get_data_by_index(val_idxs)
            test_data = self._get_data_by_index(test_idxs)
        else:
            print(f"Warning: Unknown split_segmentation mode '{self.split_segmentation}'. Returning empty.")
        if self.save_splits == "1":
            out_subdir = os.path.join(self.outdir, self.split_segmentation)
            self._save_data_split(train_data, "TrainingSet", out_subdir)
            self._save_data_split(val_data, "ValidationSet", out_subdir)
            self._save_data_split(test_data, "TestSet", out_subdir)

        return train_data, test_data, val_data

    def run(self) -> Tuple[Dict, Dict, Dict]:
        self.load_intervals()
        self.build_all_chromosomes()
        return self.get_splits()