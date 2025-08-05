import argparse
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from network_947331_freeze import Network
from spectrum_dataset import SpectrumDataset


# Environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def setup_seed(seed: int) -> None:
    """Initialize random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_model(model_dir: str) -> Network:
    """Load the best pretrained model based on validation loss.
    
    Args:
        model_dir: Directory containing pretrained models
        
    Returns:
        Initialized network with pretrained weights
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    model_files.sort(key=lambda x: float(x[14:-4]))
    best_model_path = os.path.join(model_dir, model_files[0])
    
    pretrained_model = torch.load(best_model_path)
    net = Network()
    net_dict = net.state_dict()
    
    # Filter out unnecessary pretrained weights
    pretrained_dict = {
        k: v for k, v in pretrained_model.state_dict().items() 
        if k in net_dict
    }
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    
    return net


def main():
    """Main training procedure with argument parsing."""
    # Initialize random seed
    setup_seed(2)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train network with specific random state.'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        required=True,
        help='Random state for dataset shuffling'
    )
    args = parser.parse_args()

    # Configuration constants
    BATCH_SIZE = 5
    NUM_EPOCHS = 200
    INITIAL_LR = 0.005
    PRETRAINED_DIR = "pre_weights_training"
    
    # Create output directory
    weights_dir = f'weights{args.random_state}'
    if os.path.exists(weights_dir):
        raise ValueError(
            f"Directory {weights_dir} already exists. "
            "Please choose a different random_state value."
        )
    os.makedirs(weights_dir)

    # Prepare datasets and loaders
    train_data = SpectrumDataset("train", random_state=args.random_state)
    test_data = SpectrumDataset("test", random_state=args.random_state)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize network
    net = load_pretrained_model(PRETRAINED_DIR)
    if torch.cuda.is_available():
        net.cuda()

    # Initialize optimizer and scheduler
    optimizer = Adam(net.parameters(), lr=INITIAL_LR)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        verbose=False,
        min_lr=0,
        eps=1e-08
    )

    loss_function = nn.L1Loss()
    loss_history = []

    # Training loop
    for epoch_idx in range(NUM_EPOCHS):
        print(f"Epoch {epoch_idx}, LR: {optimizer.param_groups[0]['lr']}")
        epoch_start = time.time()

        # Training phase
        net.train()
        torch.set_grad_enabled(True)
        
        for batch_idx, (imgs, _, _, _, _, mol_kinds) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                mol_kinds = mol_kinds.cuda()

            preds, _, _ = net(imgs)
            
            # Sort predictions by molecule kind
            _, sorted_indices = torch.sort(mol_kinds.squeeze())
            sorted_preds = preds[sorted_indices]

            # Calculate pairwise ranking loss
            loss = torch.tensor(0.0, device=imgs.device)
            for i in range(len(sorted_preds)):
                for j in range(i + 1, len(sorted_preds)):
                    loss += torch.relu(
                        torch.relu(sorted_preds[i]) - sorted_preds[j] + (j - i)
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print epoch statistics
        epoch_time = time.time() - epoch_start
        print(
            f"(LR:{optimizer.param_groups[0]['lr']}) "
            f"Epoch time: {epoch_time:.4f}s"
        )

        # Validation phase
        net.eval()
        torch.set_grad_enabled(False)
        val_losses = []

        for batch_idx, (imgs, _, _, _, _, mol_kinds) in enumerate(test_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                mol_kinds = mol_kinds.cuda()

            preds, _, _ = net(imgs)
            loss = loss_function(preds, mol_kinds)
            val_losses.append(loss)

        # Calculate validation metrics
        mean_loss = sum(val_losses) / len(val_losses)
        scheduler.step(mean_loss.item())

        print(
            f"Total loss: {sum(val_losses)} "
            f"Mean loss: {mean_loss:.4f}"
        )
        print(
            f"[Test] Epoch [{epoch_idx}/{NUM_EPOCHS}] "
            f"Loss: {mean_loss.item():.4f}"
        )

        # Save model and loss history
        model_path = os.path.join(
            weights_dir,
            f'net_mean_loss_{mean_loss.item():.4f}.pth'
        )
        print(f'Saving model to {model_path}\n')
        torch.save(net, model_path)

        loss_history.append(mean_loss.item())
        pd.DataFrame(loss_history).to_csv(
            os.path.join(weights_dir, 'loss_history.csv'),
            mode='a',
            index=False,
            header=False
        )

        # Clean up old model files (keep top 2)
        model_files = [
            f for f in os.listdir(weights_dir) 
            if f.endswith(".pth")
        ]
        model_files.sort(key=lambda x: float(x[14:-4]))
        
        if len(model_files) >= 3:
            for old_model in model_files[2:]:
                os.remove(os.path.join(weights_dir, old_model))


if __name__ == '__main__':
    main()

