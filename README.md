# ARIL-CPU

A CPU-optimized version of ARIL (Activity and Room-level Indoor Localization) using CSI data and Transformer networks.

## Overview

This project implements a dual-task neural network for activity and location recognition using CSI (Channel State Information) data. The model architecture combines a Transformer-based approach with efficient CPU processing, making it accessible for environments without GPU acceleration.

## Features

- Dual-task learning for both activity and location recognition
- Transformer-based architecture optimized for CPU usage
- Real-time progress tracking and visualization
- Efficient memory management
- Comprehensive metric logging and visualization

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm

## Usage

1. Prepare your CSI data in the required format
2. Run training:
   ```bash
   python3 train_cpu.py
   ```
3. For testing:
   ```bash
   python3 test_cpu.py
   ```
4. Visualize results:
   ```bash
   python3 viz_results.py
   ```

## Model Architecture

The model uses a TransformerCSI architecture with the following components:
- Input embedding layer
- Multi-head self-attention layers
- Position-wise feed-forward networks
- Dual output heads for activity and location prediction

## Directory Structure

```
.
├── models/
│   └── transformer_csi.py
├── train_cpu.py
├── test_cpu.py
├── viz_results.py
└── README.md
```

## License

This project is a modified version of the original ARIL project, adapted for CPU usage and enhanced with additional features.

