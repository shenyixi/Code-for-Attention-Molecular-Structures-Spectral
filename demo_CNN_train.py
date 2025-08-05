import torch
import torch.utils.data as data
import torch.nn as nn
import time
import random
import numpy as np
from spectrum_dataset import SpectrumDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os


# Environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Constants
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
SAVE_DIR = "/results"


def train_model():
    """Main training procedure for the neural network."""
    # Set random seed for reproducibility
    setup_seed(2)

    print('Training process started')

    # Prepare datasets
    train_data = SpectrumDataset("train")
    test_data = SpectrumDataset("test")

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize network
    net = Network()
    if torch.cuda.is_available():
        net.cuda()

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        verbose=False,
        min_lr=0,
        eps=1e-08
    )

    loss_function = nn.MSELoss()
    loss_history = []

    # Training loop
    for epoch_idx in range(NUM_EPOCHS):
        print(f"Epoch {epoch_idx}, Learning rate: {optimizer.param_groups[0]['lr']}")
        epoch_start_time = time.time()

        # Training phase
        net.train()
        torch.set_grad_enabled(True)
        for batch_idx, (img_batch, label_batch, *_) in enumerate(train_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            predict, _, _ = net(img_batch)
            loss = loss_function(predict, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print epoch duration
        epoch_time = time.time() - epoch_start_time
        print(f"(LR:{optimizer.param_groups[0]['lr']}) Epoch time: {epoch_time:.4f}s")

        # Evaluation phase
        net.eval()
        torch.set_grad_enabled(False)
        total_loss = []
        
        for batch_idx, (img_batch, label_batch, *_) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            predict, _, _ = net(img_batch)
            loss = loss_function(predict, label_batch)
            total_loss.append(loss)

        # Calculate and print metrics
        mean_loss = sum(total_loss) / len(total_loss)
        scheduler.step(mean_loss.item())

        print(f"Total loss: {sum(total_loss)} Mean loss: {mean_loss:.4f}")
        print(f"[Test] Epoch [{epoch_idx}/{NUM_EPOCHS}] Loss: {mean_loss.item():.4f}")

        # Save model
        weight_path = f'{SAVE_DIR}/netr_mean_loss_{mean_loss.item():.4f}.pth'
        print(f'Saving model to {weight_path}\n')
        torch.save(net, weight_path)

        # Save loss history
        loss_history.append(mean_loss.item())
        pd.DataFrame(loss_history).to_csv(
            'loss_history.csv',
            mode='a',
            index=False,
            header=False
        )

        # Clean up old model files
        model_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pth")]
        model_files.sort(key=lambda x: float(x[15:-4]))
        if len(model_files) >= 6:
            for old_model in model_files[5:]:
                os.remove(f"{SAVE_DIR}/{old_model}")


if __name__ == '__main__':
    train_model()

