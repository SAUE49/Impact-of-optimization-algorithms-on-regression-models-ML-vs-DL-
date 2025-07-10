# Impact of Optimization Algorithms on Regression Models

This project investigates the performance of different optimization algorithms on regression models using two popular datasets: Boston Housing and California Housing. The study compares various optimization techniques including SGD, Adam, L-BFGS, and Newton-CG across both linear regression and neural network models.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Models](#models)
- [Optimization Algorithms](#optimization-algorithms)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)

## Overview

This research project explores how different optimization algorithms affect the convergence speed, final performance, and training stability of regression models. The study provides comparative analysis through:

- **Performance Metrics**: MSE loss, R² score, convergence speed
- **Hyperparameter Analysis**: Learning rate sensitivity, momentum effects
- **Visualization**: Training/test loss curves, convergence comparisons
- **Statistical Analysis**: Epochs to convergence, final performance comparison

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
scipy
```

## Datasets

### 1. Boston Housing Dataset
- **Features**: 13 attributes (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target**: Median home value (MEDV)
- **Size**: 506 samples
- **Source**: CMU StatLib repository

### 2. California Housing Dataset
- **Features**: 8 attributes (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
- **Target**: Median house value (MedHouseVal)
- **Size**: 20,640 samples
- **Source**: Scikit-learn built-in dataset

## Models

### 1. Linear Regression Model
- Single linear layer: `nn.Linear(input_dim, 1)`
- Direct mapping from features to target value

### 2. Neural Network Model
- Architecture: Input → Hidden Layer (64 units) → ReLU → Output
- Hidden layer with ReLU activation for non-linear mapping

## Optimization Algorithms

### 1. Stochastic Gradient Descent (SGD)
- **Features**: Momentum support
- **Hyperparameters**: Learning rate, momentum coefficient
- **Characteristics**: Simple, stable, requires careful tuning

### 2. Adam Optimizer
- **Features**: Adaptive learning rates, momentum
- **Hyperparameters**: Learning rate, beta parameters
- **Characteristics**: Fast convergence, less sensitive to learning rate

### 3. L-BFGS (Limited-memory BFGS)
- **Features**: Quasi-Newton method, second-order optimization
- **Hyperparameters**: Learning rate, memory history
- **Characteristics**: Fast convergence, memory efficient

### 4. Newton-CG
- **Features**: Newton's method with conjugate gradient
- **Hyperparameters**: Maximum iterations
- **Characteristics**: Very fast convergence, requires gradient computation

## Project Structure

```
Optimsation Projet/
├── OptimsationProjet.ipynb          # Main Jupyter notebook
├── README.md                        # This file
├── Impact of optimization algorithms on regression models.docx
└── Impact of optimization algorithms on regression models.pptx
```

## Usage

1. **Environment Setup**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torch scipy
   ```

2. **Run the Notebook**:
   Open `OptimsationProjet.ipynb` in Jupyter Lab/Notebook and run all cells sequentially.

3. **Analysis Sections**:
   - **Data Loading & Preprocessing**: Load datasets and prepare train/test splits
   - **Model Definition**: Define Linear Regression and Neural Network architectures
   - **Training Functions**: Implement training loops for different optimizers
   - **Hyperparameter Analysis**: Test different learning rates and momentum values
   - **Performance Comparison**: Compare convergence speed and final performance
   - **Visualization**: Generate plots for training curves and parameter sensitivity

## Results

The project generates several types of analysis:

### 1. Training/Test Loss Curves
- Side-by-side comparison of training and test losses
- Visual assessment of convergence speed and overfitting

### 2. Convergence Speed Analysis
- Epochs required to reach specific loss thresholds
- Comparison across different optimizers and datasets

### 3. R² Score Performance
- Final model performance on test sets
- Comparison of generalization ability

### 4. Hyperparameter Sensitivity
- Learning rate vs. final loss plots
- Momentum coefficient effects (for SGD)

### 5. Comprehensive Convergence Comparison
- Multi-optimizer comparison on single plots
- Different learning rates and model architectures

## Key Findings

### Boston Housing Dataset
- **Linear Regression**: All optimizers achieve similar final performance
- **Neural Network**: L-BFGS shows fastest convergence
- **Learning Rate Sensitivity**: Adam most robust to learning rate changes

### California Housing Dataset
- **Larger Dataset Effects**: Different convergence patterns due to dataset size
- **Scalability**: Adam and SGD handle large datasets better
- **Performance**: Neural networks show improvement over linear models

### General Observations
- **Convergence Speed**: L-BFGS > Adam > SGD with momentum
- **Stability**: Adam most stable across different learning rates
- **Memory Usage**: SGD most memory efficient
- **Final Performance**: Similar final performance when properly tuned

## Future Work

- Implement additional optimizers (RMSprop, AdaGrad, etc.)
- Test on more complex neural network architectures
- Investigate learning rate scheduling effects
- Add regularization techniques comparison
- Extend to other regression datasets

## Authors

This project was developed as a **Projet Fin Module** for the **Module Analyse d'Optimisation** course.

### Students:
- **ABDESSAMAD SALLAMA**
- **YOUSSEF TAKI**
- **AMINE BAQOUCH**

### Academic Context:
- **Program**: Master Big Data & Data Science
- **Institution**: FSBM (Faculté des Sciences Ben M'Sik)
- **Course**: Module Analyse d'Optimisation

### Supervision:
- **MME BENNABOU FOUZIA**
- **MME ZINEB ELLAKY**

---

This research focuses on practical applications of optimization algorithms in machine learning regression tasks, providing comprehensive analysis of convergence behavior and performance characteristics across different optimization methods.
