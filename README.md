# Code-for-Attention-Molecular-Structures-Spectral
We propose a self-attention-based machine learning method integrating infrared/Raman spectra and molecular structures. Applied to bridged azobenzenes, it evaluates all chemical contributions, achieving accurate C-N=N-C angle predictions (r=0.99, MAE=5°).

## 1. Requirements

1.1 python 3.7.7 (>=3.8)

1.2 Other requirements
```bash
pip install -U --no-cache-dir \
    pandas==2.0.0 \
    torch==1.12.1 \
    torchvision==0.13.1
```

## 2.Train
```bash
cd code
python demo_CNN_train.py
```

## 3.Retrain
```bash
cd code
python demo_CNN_retrain.py
```

Data
sign_minmax_4400.csv: Preprocessed IR and Raman spectra used for training.

IR_Raman_azo.csv: IR and Raman spectra used for transfer learning.
