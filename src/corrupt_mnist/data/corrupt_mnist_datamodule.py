import os
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import shutil
from pathlib import Path

class CorruptMNISTDataModule(pl.LightningDataModule):
    def __init__(self, raw_dir: str, processed_dir: str, batch_size: int = 32):
        super().__init__()
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.gdrive_url = "https://drive.google.com/drive/folders/1ddWeCcsfmelqxF8sOGBihY9IU98S9JRP?usp=sharing"

    @staticmethod
    def normalize(images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        return (images - images.mean()) / images.std()

    def download_data(self):
        """Download the data from Google Drive if it doesn't exist and organize it."""
        raw_dir_absolute = os.path.abspath(self.raw_dir)

        if not os.path.exists(raw_dir_absolute):
            os.makedirs(raw_dir_absolute, exist_ok=True)

        # Check if the raw data directory contains expected files
        expected_file = os.path.join(raw_dir_absolute, "train_images_0.pt")
        if not os.path.exists(expected_file):
            print(f"Downloading data to {raw_dir_absolute}...")
            try:
                # Set the working directory to the raw_dir
                subprocess.run(
                    ["gdown", "--folder", self.gdrive_url, "-O", raw_dir_absolute],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError("Failed to download the dataset using gdown.") from e
            print("Download completed.")

            # Move or rename files if necessary
            downloaded_files = list(Path(raw_dir_absolute).glob("**/*.pt"))
            for file in downloaded_files:
                destination = Path(raw_dir_absolute) / file.name
                if not destination.exists():
                    shutil.move(str(file), str(destination))

            # Remove any extraneous subdirectories left behind
            for folder in Path(raw_dir_absolute).glob("*/"):
                if folder.is_dir():
                    shutil.rmtree(folder)

            print("Files organized.")

    def preprocess_data(self) -> None:
        """Process raw data and save it to the processed directory."""
        # self.download_data()  # Ensure data is downloaded before preprocessing

        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{self.raw_dir}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.raw_dir}/train_target_{i}.pt"))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        test_images: torch.Tensor = torch.load(f"{self.raw_dir}/test_images.pt")
        test_target: torch.Tensor = torch.load(f"{self.raw_dir}/test_target.pt")

        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_target = train_target.long()
        test_target = test_target.long()

        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(train_images, f"{self.processed_dir}/train_images.pt")
        torch.save(train_target, f"{self.processed_dir}/train_target.pt")
        torch.save(test_images, f"{self.processed_dir}/test_images.pt")
        torch.save(test_target, f"{self.processed_dir}/test_target.pt")

    def setup(self, stage: str = None):
        """Load and prepare datasets."""
        # Check if processed data exists; if not, preprocess the data
        if not os.path.exists(f"{self.processed_dir}/train_images.pt"):
            print("Processed data not found. Preprocessing data...")
            self.preprocess_data()

        if stage == "fit" or stage is None:
            train_images = torch.load(f"{self.processed_dir}/train_images.pt")
            train_target = torch.load(f"{self.processed_dir}/train_target.pt")
            self.train_set = TensorDataset(train_images, train_target)

        if stage == "test" or stage is None:
            test_images = torch.load(f"{self.processed_dir}/test_images.pt")
            test_target = torch.load(f"{self.processed_dir}/test_target.pt")
            self.test_set = TensorDataset(test_images, test_target)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

# Example usage:
# raw_dir = "data/raw"
# processed_dir = "data/processed"
# data_module = CorruptMNISTDataModule(raw_dir, processed_dir, batch_size=64)
# data_module.setup("fit")
# train_loader = data_module.train_dataloader()

