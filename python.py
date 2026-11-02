import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_excel('crop_yield.xlsx')


target = 'Production'


scaler_y = StandardScaler()
y = scaler_y.fit_transform(df[[target]]).ravel() 
X = df.drop(columns=[target])
X = pd.get_dummies(X, drop_first=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics per fold
fold_rmse = []
fold_mae = []
fold_r2 = []

print("Training Bi-layered Neural Network using 5-Fold Cross-Validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # 4. Define Bi-layered Neural Network (Two hidden layers with 64 and 32 neurons respectively)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  
        activation='relu',
        solver='adam',
        max_iter=1500,               
        learning_rate_init=0.01,     
        random_state=42
    )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_val_scaled)
    
    # Calculate fold metrics (in standardized units)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    fold_rmse.append(rmse)
    fold_mae.append(mae)
    fold_r2.append(r2)
    
    print(f"Fold {fold} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

# 5. Output Overall Validation Metrics (Should align closely with Table III)
print("\n--- Final Validation Results (Averaged Over 5 Folds) ---")
print(f"Validation RMSE: {np.mean(fold_rmse):.4f}")
print(f"Validation MAE:  {np.mean(fold_mae):.4f}")
print(f"Validation R²:   {np.mean(fold_r2):.4f}")

