# Earthquake Magnitude Prediction

This repository contains the implementation of **Earthquake Magnitude Prediction** using multiple machine learning and deep learning techniques.The aim of this project is
to build predictive models that estimate earthquake magnitudes based on various geological and geographical features.


## Features of the Repository
- Data Preprocessing: Data cleaning, handling missing values, and feature engineering (e.g., encoding categorical variables).
- Exploratory Data Analysis: Correlation analysis, heatmaps, and scatter plots.
- Machine Learning Models:
  - Random Forest Regression
  - Support Vector Regression (SVR)
  - Extreme Gradient Boosting (XGBoost)
- Clustering:
  - K-Means Clustering for identifying earthquake hotspots.
- Deep Learning Models:
  - Feedforward Neural Networks
  - LSTM (Long Short-Term Memory) Networks for time-series analysis.
- Visualization:
  - Actual vs Predicted plots
  - Residual plots
  - Geographical mappings using Folium.
  

### Prerequisites
- Python 3.8 or later
- Libraries:
  - `numpy`, `pandas`, `scikit-learn`
  - `matplotlib`, `seaborn`
  - `xgboost`, `torch`, `folium`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/earthquake-magnitude-prediction.git



Project Workflow
Data Loading: Load and explore the dataset earth.csv, containing features like latitude, longitude, depth, magnitude, and more.

Data Preprocessing:Remove duplicates.Handle missing values.Encode categorical variables like magType.

Model Training and Evaluation:Train ML models like Random Forest and XGBoost.Train neural networks (Feedforward and LSTM) using PyTorch.
Evaluate using metrics like Mean Squared Error (MSE) and R-Squared.

Hotspot Analysis:Apply K-Means clustering to identify regions with high earthquake activity.Visualize clusters using scatter plots and interactive maps.

Results
Machine Learning:
Random Forest: MSE = 22.57, R² = 79.45%
XGBoost: MSE = 17.40, R² = 86.19%
Deep Learning:
Feedforward Neural Network: MSE = 0.15, R² = 99.88%
LSTM: MSE = 23.34, R² = 82.56%

Visualization
Correlation Heatmaps
Scatter Plots for Predicted vs Actual Magnitude
Residual Analysis
Interactive Maps for Earthquake Hotspots

For questions or feedback, please contact: Name: Stuti Goel
Email: [stutigoel122@gmail.com]

