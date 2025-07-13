import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from detector import Detector
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score


class CarDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # IMPORTANT: Should match denormalization in compute_frequency_input()
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert labels to numerical values
        self.label_map = {"Fake": 0, "Real": 1}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = self.label_map[item["label"]]
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


def compute_frequency_input(image):
    # IMPORTANT: Should match normalization in CarDataset
    # Assuming image is RGB tensor [B, 3, H, W] normalized with ImageNet stats
    # Denormalize to [0,1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
    denorm_image = image * std + mean
    
    # Convert to grayscale using denormalized image
    gray = 0.299 * denorm_image[:, 0, :, :] + 0.587 * denorm_image[:, 1, :, :] + 0.114 * denorm_image[:, 2, :, :]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]
    
    # Compute FFT
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    
    # Magnitude spectrum
    magnitude = torch.log(torch.abs(fft_shift) + 1e-10)  # Log scale for better visualization/detection
    
    # Normalize to [0,1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)
    
    return magnitude

class VanillaPytorchDetector(Detector):
    def __init__(self):
        self.model = None
        self.optimizer = None

    def save_checkpoint(self, filename):
        checkpoint_dir = "checkpoints"        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename):
        checkpoint_dir = "checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _get_data_collator(self):
        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            labels = torch.stack([item["label"] for item in batch])
            return {
                "image": images,
                "label": labels
            }
        return collate_fn

    def _compute_metrics(self, predictions, true_labels):
        preds = predictions.argmax(dim=1)
        
        accuracy = accuracy_score(true_labels.cpu().numpy(), preds.cpu().numpy())
        precision = precision_score(true_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        recall = recall_score(true_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        
        # Count bad predictions
        bad_predictions = (preds != true_labels).sum().item()
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "bad_predictions": bad_predictions
        }

    def train_model(self, train_dataset, eval_dataset, num_epochs, batch_size) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        best_precision = -1.0
        best_epoch = -1
        best_checkpoint_filename = ""
        
        # Initialize model and move to device
        self.model = self.model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            CarDataset(train_dataset),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._get_data_collator()
        )
        
        eval_loader = DataLoader(
            CarDataset(eval_dataset),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._get_data_collator()
        )
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            train_metrics = []
            
            # Training
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)") as pbar:
                for batch in pbar:
                    # Move batch to device
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    metrics = self._compute_metrics(outputs, labels)
                    train_metrics.append(metrics)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "accuracy": f"{metrics['accuracy']:.4f}"
                    })
            
            # Calculate average training metrics
            avg_train_metrics = {
                k: np.mean([m[k] for m in train_metrics])
                for k in train_metrics[0].keys()
            }
            
            # Evaluation
            self.model.eval()
            eval_loss = 0
            eval_metrics = []
            
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Eval)"):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    eval_loss += loss.item()
                    metrics = self._compute_metrics(outputs, labels)
                    eval_metrics.append(metrics)
            
            avg_eval_metrics = {
                k: np.mean([m[k] for m in eval_metrics])
                for k in eval_metrics[0].keys()
            }
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Eval Loss: {eval_loss/len(eval_loader):.4f}")
            print(f"Train Metrics: {avg_train_metrics}")
            print(f"Eval Metrics: {avg_eval_metrics}\n")

            if avg_eval_metrics["precision"] > best_precision:
                best_precision = avg_eval_metrics["precision"]
                best_epoch = epoch + 1
                best_checkpoint_filename = f"{self.__class__.__name__}_epoch_{best_epoch}.pth"
                print(f"New best precision: {avg_eval_metrics['precision']:.4f}, saving checkpoint to {best_checkpoint_filename}\n")
                self.save_checkpoint(best_checkpoint_filename)

        print(f"\nBest precision: {best_precision:.4f}, at epoch {best_epoch}, loading checkpoint from: {best_checkpoint_filename}\n")
        self.load_checkpoint(best_checkpoint_filename)
            
        return self.model

    def infer_model(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            
            # Get model output
            output = self.model(image)
            
            # Get predicted class (0 or 1)
            pred_class = output.argmax(dim=1).item()
            
            # Get confidence score
            #confidence = torch.softmax(output, dim=1).max().item()
            
        return "Real" if pred_class == 1 else "Fake"
