import PIL.Image
import pandas as pd
import torch.nn as nn
from typing import Tuple, List, Dict
from sklearn.metrics import confusion_matrix
from data_loader import CarDamageDatasetLoader


class Detector:
    def get_model(self) -> nn.Module:
        pass

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass

    def train_model(self, train_dataset, eval_dataset, num_epochs, batch_size) -> int:
        pass

    def infer_model(self, image) -> str:
        pass

    def evaluate_model(self, eval_dataset) -> Tuple[List[str], List[str], List[PIL.Image.Image], pd.DataFrame, pd.DataFrame, float, int, Dict[str, List[str]]]:
        pass

    def _compute_inference_metrics(self, df_cm, y_true) -> Tuple[pd.DataFrame, float]:
        metrics = []
        total_tp_tn = 0

        for label in set(y_true):
            df_label = df_cm.loc[[label]]
            tp = df_label[label].iloc[0]
            fp = df_label.sum(axis=1).iloc[0] - tp
            fn = df_cm[label].sum() - tp
            total_tp_tn += tp

            precision = tp / (tp+fp)
            recall = tp / (tp+fn)

            metrics.append({"label": label, "precision": precision, "recall": recall})

        accuracy = (total_tp_tn / len(y_true)) * 100
        df_metrics = pd.DataFrame(metrics)
        df_metrics.set_index("label")

        return df_metrics, accuracy

    def evaluate_model(self, eval_dataset) -> Tuple[List[str], List[str], List[PIL.Image.Image], pd.DataFrame, pd.DataFrame, float, int, Dict[str, List[str]]]:
        y_true = []
        y_pred = []
        images = []
        
        bad_prediction_count = 0
        bad_predictions = {}

        for sample in eval_dataset:
            image = sample["image"]
            label = sample["label"]

            prediction = self.infer_model(image)
            y_true.append(label)
            y_pred.append(prediction)
            images.append(image)

            if not prediction in CarDamageDatasetLoader.labels:
                bad_prediction_count += 1
                if not label in bad_predictions:
                    bad_predictions[label] = [prediction]
                else:
                    bad_predictions[label].append(prediction)

        labels_set = sorted(set(y_true))
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=CarDamageDatasetLoader.labels)
        df_cm = pd.DataFrame(cm, index=CarDamageDatasetLoader.labels, columns=labels_set)

        df_metrics, accuracy = self._compute_inference_metrics(df_cm, y_true)

        return y_true, y_pred, images, df_cm, df_metrics, accuracy, bad_prediction_count, bad_predictions
