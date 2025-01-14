import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pathlib import Path
import torch
# from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

# from data.corrupt_mnist_datamodule import CorruptMNISTDataModule
# from models.models import MyAwesomeModel
# from models.corrupt_mnist_module import CorruptMNISTModel
from hydra.utils import to_absolute_path, instantiate

from data.corrupt_mnist_datamodule import CorruptMNISTDataModule


# Training function with Hydra integration
@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # wandb_logger = WandbLogger(project=cfg.project_name, log_model="all")

    # Initialize DataModule and Model
    data_module = instantiate(cfg.data_module)
    model = instantiate(cfg.model)
    wandb_logger = instantiate(cfg.logger)


    wandb_logger.watch(model, log_freq=500)

    callbacks = instantiate(cfg.callbacks)

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=wandb_logger)


    # Add a ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="train_accuracy",
        dirpath=cfg.checkpoint_dir,
        filename="corrupt_mnist-{epoch:02d}-{train_accuracy:.2f}",
        save_top_k=1,
        mode="max",
    )

    # Train the model
    # trainer = pl.Trainer(
    #     max_epochs=cfg.epochs,
    #     logger=wandb_logger,
    #     callbacks=[checkpoint_callback],
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    # )
    trainer.fit(model, data_module)

    # Finalize the W&B run

     # Log the best model as a W&B artifact
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="Trained Corrupt MNIST model",
            metadata={
                "epochs": cfg.epochs,
                "batch_size": cfg.data_module.batch_size,
                "lr": cfg.model.lr,
            },
        )
        artifact.add_file(best_model_path)
        wandb_logger.experiment.log_artifact(artifact)
        print(f"Logged model artifact: {best_model_path}")
    else:
        print("No model was saved during training.")

    wandb_logger.experiment.finish()

if __name__ == "__main__":
    train()

