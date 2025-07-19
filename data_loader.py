import os
import math
import PIL.Image
import pandas as pd
from glob import glob
from typing import Optional, Literal

class ImageSizeOptions:
    """
    Class to specify target image size and preprocessing options.
    
    Args:
        target_size: Target size as (width, height) tuple
        mode: 'resize' to resize image, 'crop' to crop image
        crop_alignment: Only applicable when mode='crop', specifies alignment:
            - 'left': Crop from left side
            - 'center': Crop from center
            - 'right': Crop from right side
    """
    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mode: Literal['resize', 'crop'] = 'resize',
        crop_alignment: Optional[Literal['left', 'center', 'right']] = 'center'
    ):
        self.target_size = target_size
        self.mode = mode
        self.crop_alignment = crop_alignment
        
        if mode == 'crop' and crop_alignment is None:
            raise ValueError("crop_alignment must be specified when mode='crop'")
        
        if mode not in ['resize', 'crop']:
            raise ValueError("mode must be either 'resize' or 'crop'")
        
        if crop_alignment and crop_alignment not in ['left_top', 'center', 'right_bottom']:
            raise ValueError("crop_alignment must be one of 'left_top', 'center', or 'right_bottom'")

    def process_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Process the image according to the specified options.
        
        Args:
            image: PIL.Image to process
            
        Returns:
            Processed PIL.Image
        """
        if self.mode == 'resize':
            return image.resize(self.target_size, PIL.Image.Resampling.BILINEAR)
            
        # Crop mode
        original_width, original_height = image.size
        target_width, target_height = self.target_size
        
        if original_width <= target_width and original_height <= target_height:
            # If image is smaller than target size, return original
            return image
            
        if self.crop_alignment == 'left_top':
            left = 0
            right = left + target_width
            top = 0
            bottom = top + target_height
        elif self.crop_alignment == 'center':
            left = (original_width - target_width) // 2
            right = left + target_width
            top = (original_height - target_height) // 2
            bottom = top + target_height
        else:  # right alignment
            left = original_width - target_width
            right = original_width
            top = original_height - target_height
            bottom = top + target_height
        
        return image.crop((left, top, right, bottom))


class CarDamageDatasetLoader:
    labels = ["Fake", "Real"]
    label_folders = {
        "Fake": "fake",
        "Real": "real"
    }

    def __init__(
        self,
        dataset_folder = os.path.join(os.path.dirname(__file__), "dataset"),
        sample_per_label = 1000,
        training_percent = 0.9,
        image_size_options = {"Fake": ImageSizeOptions(), "Real": ImageSizeOptions()}
    ):
        if not os.path.isdir(dataset_folder):
            raise Exception(f"{dataset_folder} path does not exist")

        self.image_size_options = image_size_options
        self.dataset_folder = dataset_folder
        self.sample_per_label = sample_per_label
        self.training_percent = training_percent

    def _load_image(self, image_path, label):
        image = PIL.Image.open(image_path)
        return self.image_size_options[label].process_image(image)

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

            training_data = [{"id": file, "image": self._load_image(file, label), "label": label} for file in files[:train_count]]
            eval_data = [{"id": file, "image": self._load_image(file, label), "label": label} for file in files[train_count: train_count + eval_count]]

            train_ds.extend(training_data)
            eval_ds.extend(eval_data)

        return train_ds, eval_ds, pd.DataFrame(train_ds), pd.DataFrame(eval_ds)