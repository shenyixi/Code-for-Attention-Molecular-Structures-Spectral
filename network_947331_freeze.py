import torch
import torch.nn as nn
from non_local_embedded_gaussian import NONLocalBlock1D


def freeze_layers(model: nn.Module) -> None:
    """Freeze all parameters in the given model.
    
    Args:
        model: Neural network module to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


class Network(nn.Module):
    """1D Convolutional Neural Network with Non-Local blocks for feature extraction.
    
    Architecture:
        - Three convolutional blocks with increasing channels
        - Two Non-Local attention blocks
        - Two fully-connected layers with dropout
        - Output clamped to [0, 180] range
    """

    def __init__(self) -> None:
        """Initialize network layers and freeze initial blocks."""
        super(Network, self).__init__()

        # First convolutional block (frozen)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding=4
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.non_local1 = NONLocalBlock1D(in_channels=32)
        freeze_layers(self.conv_block1)
        freeze_layers(self.non_local1)

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.non_local2 = NONLocalBlock1D(in_channels=64)

        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Fully-connected layers
        self.fc_layers1 = nn.Sequential(
            nn.Linear(in_features=64000, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc_layers2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 2, sequence_length)
            
        Returns:
            tuple: (output predictions, attention weights from first NL block,
                   attention weights from second NL block)
        """
        batch_size = x.size(0)

        # Feature extraction pathway
        features = self.conv_block1(x)
        features, attn_weights1 = self.non_local1(features)
        
        features = self.conv_block2(features)
        features, attn_weights2 = self.non_local2(features)
        
        features = self.conv_block3(features)
        features = features.view(batch_size, -1)

        # Classification pathway
        output = self.fc_layers1(features)
        output = torch.clamp(
            self.fc_layers2(output),
            min=0,
            max=180
        )

        return output, attn_weights1, attn_weights2

if __name__ == '__main__':
    main()

