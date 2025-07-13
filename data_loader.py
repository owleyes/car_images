import os
import math
import PIL.Image
import pandas as pd
from glob import glob


class CarDamageDatasetLoader:
    labels = ["Fake", "Real"]
    label_folders = {
        "Fake": "fake",
        "Real": "real"
    }

    def __init__(self, dataset_folder=os.path.join(os.path.dirname(__file__), "dataset"), sample_per_label = 1000, training_percent = 0.9):
        if not os.path.isdir(dataset_folder):
            raise Exception(f"{dataset_folder} path does not exist")

        self.size = (224, 224)
        self.dataset_folder = dataset_folder
        self.sample_per_label = sample_per_label
        self.training_percent = training_percent

    def _load_image(self, image_path):
        image = PIL.Image.open(image_path)
        
        # Resize the image to the specified size using bilinear interpolation
        image = image.resize(self.size, PIL.Image.Resampling.BILINEAR)
        return image

        # width, height = image.size
        
        # # Calculate the center crop coordinates
        # left = (width - self.size[0]) / 2
        # top = (height - self.size[1]) / 2
        # right = (width + self.size[0]) / 2
        # bottom = (height + self.size[1]) / 2
        
        # # Crop the image
        # image = image.crop((left, top, right, bottom))
        # return image

    def load_dataset(self):
        label_dict = {}
        train_ds = []
        eval_ds = []
        
        for label in self.labels:
            label_dict[label] = []
            for file in glob(os.path.join(self.dataset_folder, self.label_folders[label], "*"), recursive=True):
                label_dict[label].append(file)
            
            files = label_dict[label]
            
            train_count = int(min(
                math.floor(self.sample_per_label * self.training_percent),
                len(files) * self.training_percent
            ))

            eval_count = int(min(
                math.floor((self.sample_per_label - train_count)),
                len(files) - train_count
            ))

            training_data = [{"id": file, "image": self._load_image(file), "label": label} for file in files[:train_count]]
            eval_data = [{"id": file, "image": self._load_image(file), "label": label} for file in files[train_count: train_count + eval_count]]

            train_ds.extend(training_data)
            eval_ds.extend(eval_data)

        return train_ds, eval_ds, pd.DataFrame(train_ds), pd.DataFrame(eval_ds)