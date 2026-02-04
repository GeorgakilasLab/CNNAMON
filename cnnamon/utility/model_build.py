import json
import os
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import (confusion_matrix, # type: ignore
                             roc_curve, 
                             auc, 
                             roc_auc_score)

class ModelEvaluator:
    """

    KerasModelBuilder utility class for model evaluation purposes.
    Included functions:
    - .eval : Model evaluation based on a test set.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        class_names: list
        savefig : str Filepath to save plot.
        Returns
        -------
        Metrics : Keras evaluation metrics object

    - .cm : Confusion matrix calculation and plotting.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        savefig : str Filepath to save plot.
        Returns
        -------
        Confusion matrix : Plot

    - .roc : Reciever-Operaton Curve calculation and ploting.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        savefig : str Filepath to save plot.
        Returns
        -------
        Reciever-Operaton Curve : Plot

    - .auc : AUC calculation.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        Returns
        -------
        AUC : Keras evaluation metrics object

    """
    def __init__(self, builder):
        self.builder = builder

    def _get_predictions(self, x_test, y_test):
        model = self.builder.model
        if model is None:
            raise RuntimeError("Model not built or trained. Call train() first.")

        y_pred_raw = model.predict(x_test, verbose=0)
        output_shape = model.output_shape
        loss_config = model.loss if isinstance(model.loss, str) else model.loss.__name__
        
        y_true = y_test
        problem_type = None

        #Binary classification
        if output_shape[1] == 1:
            problem_type = 'binary'
            y_pred_labels = (y_pred_raw > 0.5).astype(int)
            y_pred_scores = y_pred_raw
        
        # Multiclass/Multilabel classification
        else:
            if 'binary_crossentropy' in str(loss_config):
                problem_type = 'multilabel'
                y_pred_labels = (y_pred_raw > 0.5).astype(int)
            else:
                problem_type = 'multiclass'
                y_pred_labels = np.argmax(y_pred_raw, axis=1)
                if y_test.ndim == 2 and y_test.shape[1] > 1:
                    y_true = np.argmax(y_test, axis=1)

            y_pred_scores = y_pred_raw

        return y_true, y_pred_labels, y_pred_scores, problem_type

    def cm(self, x_test, y_test, class_names=None, 
           title="Confusion Matrix", xlabel="Predicted", ylabel="Actual", 
           savefig=None):
        """
        .cm : Confusion matrix.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        savefig : str Filepath to save plot.
        Returns
        -------
        Confusion matrix : Plot
        """
        y_true, y_pred_labels, _, problem_type = self._get_predictions(x_test, y_test)

        if problem_type == 'multilabel':
            print("Warning: Standard CM not available for multilabel. Plotting first label only.")
            y_true = y_true[:, 0]
            y_pred_labels = y_pred_labels[:, 0]
            if class_names: class_names = ["Not " + class_names[0], class_names[0]]

        cm_raw = confusion_matrix(y_true, y_pred_labels)
        cm_norm = cm_raw.astype('float') / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-10)
        annot_labels = []
        for prop, count in zip(cm_norm.flatten(), cm_raw.flatten()):
            annot_labels.append(f"{prop:.2f}\n({count})")
        annot_labels = np.asarray(annot_labels).reshape(cm_norm.shape)
        if class_names is None:
            class_names = [str(i) for i in range(cm_raw.shape[0])]
        if len(class_names) != cm_raw.shape[0]:
            print(f"Warning: class_names length ({len(class_names)}) should match distinct labels ({cm_raw.shape[0]}).")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,vmin=0,vmax=1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        
        if savefig:
            self._save_plot(savefig)
        else:
            plt.show()

    def roc(self, x_test, y_test, class_names=None, 
            title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate",
            savefig=None):
        """
        .roc : Reciever-Operaton Curve calculation and ploting.
        Parameters
        ----------
        dataset : dict{"x":sequence tensor,"y":label tensor}
        savefig : str Filepath to save plot.
        Returns
        -------
        Reciever-Operaton Curve : Plot
        """
        y_true, _, y_pred_scores, problem_type = self._get_predictions(x_test, y_test)
        
        if problem_type != 'binary':
            print(f"Warning: ROC is for binary classification. Detected '{problem_type}'. Skipping.")
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
        auc_score = auc(fpr, tpr)
        label_text = f'AUC = {auc_score:.4f}'
        if class_names and len(class_names) == 2:
            label_text = f"{class_names[1]} vs {class_names[0]} (AUC = {auc_score:.4f})"

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='orange', lw=2, label=label_text)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="lower right")
        
        if savefig:
            self._save_plot(savefig)
        else:
            plt.show()

    def auc(self, x_test, y_test):
        y_true, _, y_pred_scores, problem_type = self._get_predictions(x_test, y_test)
        score = None
        if problem_type == 'binary':
            score = roc_auc_score(y_true, y_pred_scores)
            print(f"AUC Score: {score:.4f}")
        elif problem_type == 'multiclass':
            score = roc_auc_score(y_true, y_pred_scores, multi_class='ovr')
            print(f"AUC Score (One-vs-Rest): {score:.4f}")
        else:
            print("Warning: AUC score for multilabel. Skipping.")
        return score

    def _save_plot(self, savefig_path):
        save_dir = os.path.dirname(savefig_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(savefig_path)
        print(f"Plot saved to {savefig_path}")


class KerasModelBuilder:
    """
    Build a Keras model from JSON. The JSON file should have a specific form that includes Keras' layers, compiler, training parameters and callbacks.

    {
        "model": {
            "layers": [
                {"class_name": "Conv1D", "config": { ... }},
                {"class_name": "Dense", "config": { ... }}
            ]
        },
        "compile": {
             "optimizer": { "class_name": "Adam", "config": {"learning_rate": 0.001} },
             "loss": "binary_crossentropy",
             "metrics": ["accuracy"]
        },
        "training_params": {
            "epochs": 50,
            "batch_size": 32
        },
        "callbacks": {
            "EarlyStopping": { "monitor": "val_loss", "patience": 5 },
            "ReduceLROnPlateau": { "factor": 0.5 }
        }
    }
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.history = None
        self.eval = ModelEvaluator(self)
 
    def build(self):
        model_cfg = self.config.get("model", {})
        layers_cfg = model_cfg.get("layers", [])

        self.model = keras.Sequential()
        for layer_def in layers_cfg:
            class_name = layer_def.get("class_name")
            config = layer_def.get("config", {})

            if not hasattr(keras.layers, class_name):
                raise ValueError(f"Unknown Keras layer: {class_name}")
            
            layer_cls = getattr(keras.layers, class_name)
            self.model.add(layer_cls(**config))
        compile_cfg = self.config.get("compile", {})
        if not compile_cfg:
            raise ValueError("JSON must contain a 'compile' section.")
        optimizer_data = compile_cfg.get("optimizer", "adam")
        if isinstance(optimizer_data, dict):
            opt_name = optimizer_data.get("class_name", "Adam")
            opt_conf = optimizer_data.get("config", {})
            if hasattr(keras.optimizers, opt_name):
                optimizer_instance = getattr(keras.optimizers, opt_name)(**opt_conf)
                compile_cfg["optimizer"] = optimizer_instance
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")
        
        self.model.compile(**compile_cfg)
        return self.model

    # Training
    def train(self, x_train, y_train, x_val=None, y_val=None):
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        train_params = self.config.get("training_params", {}).copy()
        callbacks_list = []
        callbacks_cfg = self.config.get("callbacks", {})
        
        for name, params in callbacks_cfg.items():
            if hasattr(keras.callbacks, name):
                cb_cls = getattr(keras.callbacks, name)
                if name == "CSVLogger" and "filename" in params:
                     os.makedirs(os.path.dirname(params["filename"]), exist_ok=True)
                callbacks_list.append(cb_cls(**params))
            else:
                print(f"Warning: Callback '{name}' not found in keras.callbacks. Skipping.")

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            callbacks=callbacks_list,
            **train_params
        )
        self.history = history
        return history

    def evaluate(self, x_test, y_test):
        if self.model is None:
            raise RuntimeError("Model not built.")
        return self.model.evaluate(x_test, y_test, verbose=1)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.model.save(path)

    @staticmethod
    def load(path: str):
        model = keras.models.load_model(path)
        return model
    
    def summary(self):
        if self.model: self.model.summary()

    def plot_history(self, title="Training History", savefig=None):
        if self.history is None:
            raise RuntimeError("No history found.")
        
        history_dict = self.history.history
        keys = [k for k in history_dict.keys() if not k.startswith('val_')]
        metric_pairs = []
        for k in keys:
            if f"val_{k}" in history_dict:
                metric_pairs.append((k, f"val_{k}"))
            else:
                metric_pairs.append((k, None))

        plt.figure(figsize=(6 * len(metric_pairs), 5))
        epochs = range(1, len(history_dict[keys[0]]) + 1)

        for i, (metric, val_metric) in enumerate(metric_pairs):
            plt.subplot(1, len(metric_pairs), i + 1)
            plt.plot(epochs, history_dict[metric], '-', color='navy', label=f'Training {metric}')
            if val_metric:
                plt.plot(epochs, history_dict[val_metric], '-', color='orange', label=f'Validation {metric}')
            
            plt.title(f"{title}: {metric}")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend()
            
        plt.tight_layout()
        if savefig:
            self.eval._save_plot(savefig)
        else:
            plt.show()

    @classmethod
    def from_json(cls, json_path_or_dict):
        if isinstance(json_path_or_dict, str):
            with open(json_path_or_dict, "r") as f:
                config = json.load(f)
        else:
            config = json_path_or_dict
        builder = cls(config)
        builder.build()
        return builder