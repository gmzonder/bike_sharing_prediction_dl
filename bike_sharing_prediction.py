'''
A module to collect useful functions we reuse in various notebooks
'''

# We will reuse this function. Hence we copy it to bike_sharing_prediction.py

# I utilized most of code from mads_dl.py and flight_forecasting.py, 
# adapting and modifying some parts to integrate them into bike_sharing_prediction.py.

def count_params(model):
    '''
    Return the number of trainable parameters of a PyTorch Module (model)
    Iterate each of the modules parameters and counts them 
    if they require grad (if they are trainable)
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)                                               

########################################################################################

from sklearn import metrics
import torch

def compute_acc(model, X, y):
    '''
    compute the accuracy of a model for given features X and labels y
    '''
    model.eval()
    with torch.no_grad():
        y_pred=model.predict(X)
    return metrics.accuracy_score(y, y_pred)


########################################################################################

def predict(model, X):
    '''
    Use the model to predict for the values in the test set.
    Return the prediction
    '''
    model.eval()
    with torch.no_grad():
        return model(X)

########################################################################################

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import math

def add_regression_eval(results, algorithm, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, num_params):
    '''
    Create a table with evaluation results of a regression experiment
    including train, validation, and test metrics.
    '''
    # Loop through each dataset and add corresponding metrics
    for dataset, actual, predicted in zip(
        ("train", "validation", "test"), 
        (y_train, y_val, y_test), 
        (y_train_pred, y_val_pred, y_test_pred)
    ):  
        mse = mean_squared_error(actual, predicted)
        results = pd.concat([results, pd.DataFrame([{
            "algorithm": algorithm, 
            "dataset": dataset,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "MAE": mean_absolute_error(actual, predicted),
            "MAPE": mean_absolute_percentage_error(actual, predicted) * 100,  # Convert MAPE to percentage
            "params": num_params
        }])], ignore_index=True)
    
    return results



# def add_regression_eval(results, algorithm, y_train, y_train_pred, y_test, y_test_pred, num_params):
#     '''
#     Create a table with evaluation results
#     of a regression experiment
#     '''
#     for dataset, actual, predicted in zip(("train", "test"), (y_train, y_test), (y_train_pred, y_test_pred)):
#         results= pd.concat([results, pd.DataFrame([{
#             "algorithm": algorithm, 
#             "dataset": dataset,
#             "MSE": mean_squared_error(actual, predicted),
#             "MAE": mean_absolute_error(actual, predicted),
#             "MAPE": mean_absolute_percentage_error(actual, predicted)*100, # implemented is relative to 1 not to 100
#             "params": num_params
#         }])], ignore_index=True)   
#     return results

########################################################################################

# Train a model with early stopping
import copy
def train(model, num_epochs, learning_rate, loss_function, trainX, trainY, valX, valY, patience=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        trainY_pred = model(trainX)
        train_loss = loss_function(trainY_pred, trainY)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valY_pred = model(valX)
            val_loss = loss_function(valY_pred, valY)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    return model

########################################################################################

# Use model to predict for test set
def predict(model, testX):
    model.eval()
    with torch.no_grad():
        return model(testX)
    
########################################################################################

# Train a model with early stopping
import copy

def find_best_dropout(model_class, input_size, hidden_size, train_loader, val_loader, num_epochs, learning_rate, dropout_rates, patience=100):
    '''
    Find the best dropout rate from a list of rates based on validation loss.

    Parameters:
    - model_class: the class of the model (e.g., MLPWithDropout or your custom model)
    - input_size: input size for the model
    - hidden_size: hidden layer size for the model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - num_epochs: number of training epochs
    - learning_rate: learning rate for the optimizer
    - dropout_rates: list of dropout rates to test
    - patience: patience for early stopping

    Returns:
    - best_dropout_rate: the optimal dropout rate based on validation loss
    - best_model: the model trained with the optimal dropout rate
    - best_val_loss: the validation loss achieved with the optimal dropout rate
    '''
    best_dropout_rate = None
    best_val_loss = float("inf")
    best_model = None
    
    for dropout_rate in dropout_rates:
        model = model_class(input_size=input_size, hidden_size=hidden_size, dropout_prob=dropout_rate)
        model, val_loss = train_with_early_stopping(model, train_loader, val_loader, num_epochs, learning_rate, patience)

        print(f"Dropout Rate: {dropout_rate}, Validation Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dropout_rate = dropout_rate
            best_model = copy.deepcopy(model)

    print(f"Optimal Dropout Rate: {best_dropout_rate}")
    return best_dropout_rate, best_model, best_val_loss


########################################################################################
# Train a model for early stopping
import copy
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, num_epochs, learning_rate, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            # Get the expected input size by inspecting the model's first layer dynamically
            input_features = batch_X.shape[1]  # Assuming batch_X has shape [batch_size, input_size]

            optimizer.zero_grad()
            trainY_pred = model(batch_X)
            loss = criterion(trainY_pred, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average train loss
        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                valY_pred = model(batch_X)
                loss = criterion(valY_pred, batch_Y)
                val_loss += loss.item()

        # Average val loss
        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_loss


########################################################################################

# Baselines

import numpy as np
# Predict the average of the most recent n elements of x
def baseline_avg_prev_n_month(x, n):
    return torch.Tensor(np.sum(x[:,-n:], axis=1)/n)

# Predict the value of last year (12 month ago)
def baseline_last_year(x):
    return torch.Tensor(x[:,-12])

# Predict 0 as new value
def baseline_zero(x, scaler):
    return torch.Tensor(np.full((len(x),1), scaler.transform(np.array(0).reshape(-1,1))))

########################################################################################

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import copy
from sklearn.metrics import mean_absolute_percentage_error

# Count the number of trainable parameters of model
def count_params(model):
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0


# Add evaluation measures to the results DataFrame
def add_result(result_df, predictions, model, model_name, criterion, scaler, testY_pred, testY):
    loss = criterion(testY_pred, testY)
    result_df.at[model_name, "Test-Loss"] = loss.item()

    # Inverse transform predictions and actual values
    testY_pred_rescaled = scaler.inverse_transform(testY_pred.reshape(-1, 1))
    testY_rescaled = scaler.inverse_transform(testY.reshape(-1, 1))

    abs_diff = np.sum(abs(testY_pred_rescaled - testY_rescaled))
    result_df.at[model_name, "Avg. daily bike sharing deviation"] = round(abs_diff / len(testY_rescaled), 1)
    result_df.at[model_name, "Parameters"] = count_params(model)

    prediction_df = pd.DataFrame(testY_pred_rescaled, columns=[model_name])
    if predictions.empty:
        predictions.reindex_like(prediction_df)
        predictions["Actual"] = pd.DataFrame(testY_rescaled, columns=["actual"])["actual"]
    predictions[model_name] = prediction_df[model_name]

########################################################################################

# # Train a hybrid model with two inputs (features and workingday)
# def hyb_train(model, num_epochs, learning_rate, loss_function, trainX1, trainX2, trainY, valX1, valX2, valY, patience=500):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     best_val_loss = float("inf")
#     best_model_state = copy.deepcopy(model.state_dict())
#     epochs_no_improve = 0

#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         trainY_pred = model(trainX1, trainX2)
#         train_loss = loss_function(trainY_pred, trainY)
#         train_loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             valY_pred = model(valX1, valX2)
#             val_loss = loss_function(valY_pred, valY)
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_state = copy.deepcopy(model.state_dict())
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break

#     model.load_state_dict(best_model_state)
#     return model

# # Use model to predict for test set with two inputs
# def hyb_predict(model, testX1, testX2):
#     model.eval()
#     with torch.no_grad():
#         return model(testX1, testX2)

########################################################################################

# Sliding window function for time series data
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

########################################################################################

# Experimental setup for machine learning (scale data, create sliding windows, split data)
def setup_experiment(dataset, seq_len, test_share, val_share, scaler, ignored_last_month):
    scaled_data = scaler.fit_transform(dataset)
    x, y = sliding_windows(scaled_data, seq_len)
    used_length = len(y) - ignored_last_month  # Ignore the last few months if specified
    test_size = int(used_length * test_share)
    val_size = int(used_length * val_share)
    train_size = used_length - test_size - val_size

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size]
    x_test, y_test = x[train_size + val_size:used_length], y[train_size + val_size:used_length]

    return x_train, y_train, x_val, y_val, x_test, y_test

