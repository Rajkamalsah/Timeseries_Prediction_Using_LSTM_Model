# Time Series OHLC Prediction of Nifty Bank

This project aims to predict the next day's closing price of the Nifty Bank index using historical OHLC (Open, High, Low, Close) data. The model is built using LSTM (Long Short-Term Memory) neural networks and optimized using Grid Search for hyperparameter tuning.

## Features

- **Data Collection**: Download historical OHLC data for Nifty Bank using Yahoo Finance.
- **Data Preprocessing**: Scale the data and create features and labels for the model.
- **Model Building**: Implement an LSTM model using TensorFlow and Keras.
- **Hyperparameter Tuning**: Optimize the model using Grid Search with cross-validation.
- **Model Evaluation**: Evaluate the model's performance on test data.
- **Visualization**: Plot the actual vs predicted values.

## Tools and Technologies

- **Python**
- **NumPy**
- **Pandas**
- **Yahoo Finance (yfinance)**
- **Matplotlib**
- **Scikit-learn**
- **TensorFlow**
- **Keras**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/nifty-bank-prediction.git
    cd nifty-bank-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install numpy pandas yfinance matplotlib scikit-learn tensorflow
    ```

## Usage

1. **Data Collection**:
    ```python
    import yfinance as yf

    # Download data
    data = yf.download(tickers='^NSEBANK', start='2012-03-11', end='2022-07-10')
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Volume', 'Date'], axis=1, inplace=True)

    data['next_close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    ```

2. **Data Preprocessing**:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd

    # Define input features and output
    X = data[['Open', 'High', 'Low', 'Close']]
    y = data[['next_close']]

    # Scale the dataset
    sc_X = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = sc_X.fit_transform(X)
    y_scaled = sc_y.fit_transform(y)

    # Create features and labels
    X = []
    backcandles = 5
    for i in range(backcandles, len(X_scaled)):
        X.append(X_scaled[i-backcandles:i])
    X = np.array(X)
    y = y_scaled[backcandles:]

    # Split data into train and test
    splitlimit = int(len(X) * 0.8)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    ```

3. **Model Building and Training**:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Activation, Input
    from tensorflow.keras import optimizers
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.metrics import make_scorer, mean_squared_error

    # Custom Keras Regressor
    class KerasRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, units=50, learning_rate=0.001, epochs=30, batch_size=15):
            self.units = units
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def build_model(self):
            lstm_input = Input(shape=(backcandles, X.shape), name='lstm_input')
            inputs = LSTM(self.units, name='first_layer')(lstm_input)
            inputs = Dense(1, name='dense_layer')(inputs)
            output = Activation('linear', name='output')(inputs)
            model = Model(inputs=lstm_input, outputs=output)
            adam = optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=adam, loss='mse')
            return model

        def fit(self, X, y):
            self.model = self.build_model()
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict(self, X):
            return self.model.predict(X)

    # Define the grid of hyperparameters to search
    param_grid = {
        'units': [50, 100],
        'learning_rate': [0.01, 0.001],
        'batch_size': [10, 20],
        'epochs': [30, 50]
    }

    # Perform grid search
    grid = GridSearchCV(estimator=KerasRegressor(), param_grid=param_grid, cv=3, scoring=make_scorer(mean_squared_error, greater_is_better=False))
    grid_result = grid.fit(X_train, y_train)

    # Print the best parameters and best score
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

    # Train the model with the best parameters
    best_model = grid_result.best_estimator_
    best_model.fit(X_train, y_train)
    ```

4. **Prediction and Evaluation**:
    ```python
    # Prediction part
    y_pred_scaled = best_model.predict(X_test)

    # Inverse transform the scaled predictions and test values
    y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test = sc_y.inverse_transform(y_test)

    # Create a DataFrame with actual and predicted values
    results = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten()
    })

    # Calculate the difference between actual and predicted values
    results['Difference'] = results['Actual'] - results['Predicted']

    # Determine if the prediction is right (difference < 10)
    results['Right_Prediction'] = results['Difference'].abs() < 10

    # Calculate the accuracy of the predictions
    accuracy = results['Right_Prediction'].mean() * 100

    # Print the results and accuracy
    print(results.head(10))
    print(f"Accuracy of the predictions: {accuracy:.2f}%")
    ```

5. **Visualization**:
    ```python
    import matplotlib.pyplot as plt

    # Plot the actual difference values
    plt.figure(figsize=(16,8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(y_pred, color='green', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('High-Low Difference')
    plt.title('Actual vs Predicted High-Low Difference')
    plt.legend()
    plt.show()
    ```


