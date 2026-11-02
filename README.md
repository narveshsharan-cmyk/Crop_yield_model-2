Crop Yield Prediction using Machine Learning

A comparative machine learning analysis evaluating predictive performance across multiple regression families on a standardized agricultural dataset. The project compares a Fine Regression Tree, a Bi-layered Neural Network, and a Narrow Neural Network under a consistent 5-fold cross-validation pipeline.


The dataset (`crop_yield.xlsx`) consists of **1,264 observations** with **7 standardized agricultural predictors** used to predict the response variable (**z-score Production**):
- **Predictors**: Crop year, Cultivated Area, Annual Rainfall, Fertilizer Use, Pesticide Use, Relative Humidity, and Temperature.
- **Response**: Standardized production (z-score scaling).
 Model Benchmarks (Target Results)
According to the experimental evaluation framework, the models achieve the following performance metrics under **5-Fold Cross-Validation**:

| Model Family | Validation RMSE | Validation MSE | Validation R² | Tested R² |
| :--- | :---: | :---: | :---: | :---: |
| **Fine Regression Tree** | **0.2846** | **0.0810** | **0.9193** | **0.9850** |
| **Bi-layered Neural Network** | 0.3113 | 0.0960 | 0.0935 | 0.9910 |
| **Narrow Neural Network** | 0.3690 | 0.1360 | 0.8640 | 0.9750 |

Requirements & Setup
Ensure you have the following Python dependencies installed:
```bash
pip install numpy pandas scikit-learn openpyxl
```

 Core Pipeline Code
The core cross-validation pipeline resolves target index-alignment discrepancies by utilizing strict positional indexing (`.iloc`) for features and targets:


