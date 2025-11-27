Markdown

# Loss Landscape Geometry & Optimization Dynamics

## Overview
This repository implements a rigorous framework for analyzing neural network loss landscape geometry. It explores the relationship between "flat minima" and model generalization using 1D linear interpolation and 2D contour visualization techniques.

## Key Features
* **Filter Normalization:** Implements scale-invariant random projections to accurately visualize high-dimensional weight spaces.
* **Landscape Probing:** Tools to generate 1D loss curves and 2D loss surfaces around a trained model's parameters.
* **PyTorch Implementation:** Modular code structure for training and analysis.

## Project Structure
```text
loss-landscape-dynamics/
├── src/
│   ├── models.py          # CNN Architecture definition
│   └── landscape_utils.py # Math for Filter-Normalized Projections
├── train.py               # Script to train the base model
├── analyze_landscape.py   # Script to generate 1D/2D visualizations
├── requirements.txt       # Dependencies
└── results/               # Generated Plots
How to Run
Install Dependencies:

Bash

pip install -r requirements.txt
Train the Model: This trains a simple CNN on MNIST and saves the weights to trained_model.pth.

Bash

python -m src.train
Generate Landscapes: This performs the geometric analysis and saves the plots to the root directory.

Bash

python analyze_landscape.py
