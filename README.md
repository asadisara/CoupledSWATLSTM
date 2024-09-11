# CoupledSWATLSTM
# SWAT+ model outputs used as input for LSTM model to improve streamflow estimation through coupled modeling
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial.polynomial import Polynomial
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import shap
import pkg_resources

# Load and preprocess the data
data = pd.read_csv('data_SWAT_Calibrated_3001.csv')  # Replace with the actual combined data file
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
   
# Prepare the input and output columns
X_columns = ['flow_out', 'P(t)', 'Tmax(t)', 'Tmin(t)', 'surq_gen', 'latq', 'perc', 'et', 'sw_final', 'pet']
y_column = 'ObservedQ(t)'

# Scale the features using all available data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(data[X_columns].values)
y_scaled = scaler_y.fit_transform(data[y_column].values.reshape(-1, 1)).flatten()

# Split the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, shuffle=False)  # 0.1765 is approximately 15% of 0.85

# Reshape data for LSTM (samples, time steps, features)
X_train_val = X_train_val.reshape((X_train_val.shape[0], 1, X_train_val.shape[1]))
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the K-Fold cross-validator
k = 5  # Number of folds
kf = KFold(n_splits=k)

# Initialize variables to store metrics for each fold
fold_metrics = []

# Perform k-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Training on fold {fold+1}...")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_fold.shape[1], X_train_fold.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=16, verbose=0)

    # Predict using the trained model
    y_pred_train_fold = model.predict(X_train_fold).flatten()
    y_pred_val_fold = model.predict(X_val_fold).flatten()

    # Inverse transform to get original scale
    y_train_fold = scaler_y.inverse_transform(y_train_fold.reshape(-1, 1)).flatten()
    y_val_fold = scaler_y.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
    y_pred_train_fold = scaler_y.inverse_transform(y_pred_train_fold.reshape(-1, 1)).flatten()
    y_pred_val_fold = scaler_y.inverse_transform(y_pred_val_fold.reshape(-1, 1)).flatten()

    # Calculate metrics for the train period of the current fold
    train_metrics = {
        'NSE': 1 - (sum((y_train_fold - y_pred_train_fold)**2) / sum((y_train_fold - y_train_fold.mean())**2)),
        'RMSE': np.sqrt(mean_squared_error(y_train_fold, y_pred_train_fold)),
        'MAE': np.mean(np.abs(y_train_fold - y_pred_train_fold)),
        'PBIAS': 100 * sum(y_pred_train_fold - y_train_fold) / sum(y_train_fold)
        }

    # Calculate metrics for the validation period of the current fold
    val_metrics = {
        'NSE': 1 - (sum((y_val_fold - y_pred_val_fold)**2) / sum((y_val_fold - y_val_fold.mean())**2)),
        'RMSE': np.sqrt(mean_squared_error(y_val_fold, y_pred_val_fold)),
        'MAE': np.mean(np.abs(y_val_fold - y_pred_val_fold)),
        'PBIAS': 100 * sum(y_pred_val_fold - y_val_fold) / sum(y_val_fold)
        }

    # Append validation metrics to the fold metrics list
    fold_metrics.append({
        'Fold': fold+1,
        'Train_NSE' : train_metrics['NSE'],
        'Train_RMSE' : train_metrics['RMSE'],
        'Train_MAE' : train_metrics['MAE'],
        'Train_PBIAS' : train_metrics['PBIAS'],
        'Val_NSE' : val_metrics['NSE'],
        'Val_RMSE' : val_metrics['RMSE'],
        'Val_MAE' : val_metrics['MAE'],
        'Val_PBIAS' : val_metrics['PBIAS']
        })

# Convert the list of metrics to a DataFrame
fold_metrics_df = pd.DataFrame(fold_metrics)
print(fold_metrics_df)

# Save the fold metrics to a CSV file
fold_metrics_df.to_csv('5-fold_64-unit_metrics_70_15_15_lstm_3001_Calibrated.csv', index=False)

# Train the final model on the entire training and validation set
final_model = Sequential()
final_model.add(LSTM(64, input_shape=(X_train_val.shape[1], X_train_val.shape[2])))
final_model.add(Dense(1))
final_model.compile(loss='mean_squared_error', optimizer='adam')
final_model.fit(X_train_val, np.concatenate((y_train, y_val)), epochs=100, batch_size=16, verbose=0)

# Predict using the final trained model
y_pred_test = final_model.predict(X_test).flatten()

# Inverse transform to get original scale
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Calculate metrics for the test set
test_metrics = {
    'NSE': 1 - (sum((y_test - y_pred_test)**2) / sum((y_test - y_test.mean())**2)),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'MAE': np.mean(np.abs(y_test - y_pred_test)),
    'PBIAS': 100 * sum(y_pred_test - y_test) / sum(y_test)
    }

# Print and save the test metrics
print('Test Metrics:', test_metrics)

# Convert the test metrics to a DataFrame
test_metrics_df = pd.DataFrame([test_metrics], index=['Test'])

# Save the test metrics to a CSV file
test_metrics_df.to_csv('5-fold_64-unit_test_metrics_70_15_15_lstm_3001_Calibrated.csv', index=True)


# Predict using the final trained model for the entire dataset
y_pred_full = final_model.predict(X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))).flatten()

# Inverse transform to get original scale for the full dataset predictions
y_pred_full_original_scale = scaler_y.inverse_transform(y_pred_full.reshape(-1, 1)).flatten()

# Create a DataFrame with the date index, observed and predicted values
results_df = pd.DataFrame({
'Date': data.index,
'ObservedQ': scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten(),
'PredictedQ': y_pred_full_original_scale
}).set_index('Date')

# Save the results to an Excel file
results_df.to_excel('20240611_predicted_values_full_time_period_station3001_calibrated.xlsx')


#Create an Explainer
explainer = shap.DeepExplainer(final_model, X_train_val)

#Generate SHAP values
shap_values = explainer.shap_values(X_train_val)


for i, shap_array in enumerate(shap_values):
    print(f"Shape of SHAP values for class {i}: {shap_array.shape}")

print(X_train_val.shape)

# Define your feature names
feature_names = ['flow_out', 'P(t)', 'Tmax(t)', 'Tmin(t)', 'surq_gen', 'latq', 'perc', 'et', 'sw_final', 'pet']

# Flatten the SHAP values and X_train data
shap_values_flattened = shap_values[0].reshape(shap_values[0].shape[0], -1)
X_train_val_flattened = X_train_val.reshape(X_train_val.shape[0], -1)

# Generate the SHAP summary plot for your regression model
shap.summary_plot(shap_values_flattened, X_train_val_flattened, feature_names=feature_names)

# Calculate the mean absolute SHAP values for each feature
mean_shap_values = np.abs(shap_values_flattened).mean(axis=0)
# Calculate the standard deviation of the SHAP values for each feature
std_shap_values = np.abs(shap_values_flattened).std(axis=0)

# Sort the features by mean importance
sorted_indices = np.argsort(mean_shap_values)[::-1]
sorted_feature_names = np.array(feature_names)[sorted_indices]
sorted_mean_shap_values = mean_shap_values[sorted_indices]
sorted_std_shap_values = std_shap_values[sorted_indices]

# Convert mean SHAP values to percentages
total = sorted_mean_shap_values.sum()
sorted_mean_shap_values_percent = 100 * sorted_mean_shap_values / total

# Create the bar plot with error bars
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_feature_names)), sorted_mean_shap_values_percent, color=(0.561, 0.667, 0.863), capsize=3)
plt.errorbar(sorted_mean_shap_values_percent, range(len(sorted_feature_names)), xerr=sorted_std_shap_values, fmt='none', ecolor='black', elinewidth=1, capthick=1, capsize=5)
plt.xlabel('Mean Absolute SHAP Value (Importance %)', fontsize=16)  # Adjust font size

# Add the importance percentage next to each bar, shifted to the right for clarity, and in dark red color
for bar in bars:
    plt.text(bar.get_width() + max(sorted_std_shap_values) + 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', va='center', color=(1,0,0.25), fontsize=14)

# Adjust the starting position of the horizontal axis
plt.xlim(left=-2, right=max(sorted_mean_shap_values_percent) + max(sorted_std_shap_values) + 4)

plt.yticks(range(len(sorted_feature_names)), sorted_feature_names, fontsize=16)
plt.xticks(fontsize=14)
plt.xlabel('Mean Absolute SHAP Value (Importance %)', fontsize=16)
plt.title('Global Feature Importance_station 3001', fontsize=20, loc='left')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.tight_layout()  # Adjust the layout to fit all elements
plt.savefig('20240610_GlobalFeatureImp_Calibrated_station3001.png', dpi=300)  # Save the first figure with high resolution
plt.show()
