# Code-for-Attention-Molecular-Structures-Spectral
We propose a self-attention-based machine learning method integrating infrared/Raman spectra and molecular structures. Applied to bridged azobenzenes, it evaluates all chemical contributions, achieving accurate C-N=N-C angle predictions (r=0.99, MAE=5°).

## 1.Repository Structure
demo_CNN_train.py: Main script for training the model. It loads the training data, initializes the network, and trains the model. The trained model is saved as net_mean_loss_58.6649.pth.

demo_CNN_retrain.py: Script for retraining or fine-tuning the model using transfer learning. It loads a pre-trained model and retrains it on new data.

network.py: Defines the neural network architecture used during the initial training phase.

network_947331_freeze.py: Defines the neural network architecture used during transfer learning. Certain layers may be frozen to retain learned features.

non_local_embedded_gaussian.py: Implements the self-attention mechanism (non-local embedded Gaussian) used in the network.

spectrum_dataset.py: Handles data loading and preprocessing. It loads the spectral data from sign_minmax_4400.csv (training data) and IR_Raman_azo.csv (transfer learning data).

sign_minmax_4400.csv: Contains the preprocessed training data (IR and Raman spectra).

IR_Raman_azo.csv: Contains the data used for transfer learning.

net_mean_loss_58.6649.pth: Pre-trained model file saved after training.

## 2. Requirements

2.1 python 3.7.7 (>=3.8)

2.2 Other requirements
```bash
pip install -U --no-cache-dir \
    pandas==2.0.0 \
    torch==1.12.1 \
    torchvision==0.13.1
```

## 3.Train
```bash
cd code
python demo_CNN_train.py
```

## 4.Retrain
```bash
cd code
python demo_CNN_retrain.py
```
