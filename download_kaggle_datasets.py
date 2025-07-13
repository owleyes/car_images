import kagglehub


def _download_kaggle_dataset(dataset_name):
    path = kagglehub.dataset_download(dataset_name)
    print(f"Downloaded {dataset_name} to {path}")
    return path

def download_kaggle_datasets():
    _download_kaggle_dataset("pacificrm/car-insurance-fraud-detection")
    _download_kaggle_dataset("vinayjose/car-damage-dataset")

if __name__ == "__main__":
    download_kaggle_datasets()
    