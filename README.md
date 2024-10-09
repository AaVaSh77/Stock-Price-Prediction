# Stock Price Prediction using LSTM

This project demonstrates the use of a **Long Short-Term Memory (LSTM)** neural network to predict Google stock prices based on historical data. The datasets `trainset.csv` and `testset.csv` are used for training and testing the model, respectively.

### Dataset
The dataset contains the following features:
- **Date**: The date of each stock entry.
- **Open**: The opening stock price of Google on the given date.

Both the training set (`trainset.csv`) and the test set (`testset.csv`) include historical stock prices, which are used to train the LSTM model and predict future stock prices.

### 1. Data Preprocessing
- **Combining Datasets**: The training and test sets are combined to create a full dataset of stock prices.
- **Normalization**: The stock prices are normalized using the `MinMaxScaler` to scale the values between 0 and 1, which helps the model converge faster and improves performance.
- **Creating Sequences**: To capture temporal dependencies, a sliding window of 60 previous time steps (days) is used to predict the next stock price. This allows the model to learn patterns in sequential data.

### 2. Building the LSTM Model
The model is built using the following architecture:
- **LSTM Layers**: Four LSTM layers with 50 units each. The first three LSTM layers return sequences to the next LSTM layer, while the final LSTM outputs directly to the Dense layer.
- **Dropout Layers**: After each LSTM layer, a **Dropout** layer with a rate of 0.2 is applied to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Dense Layer**: A fully connected **Dense** layer with one unit is used to output the predicted stock price.

### 3. Model Training
- The model is compiled using the **Adam optimizer** and **Mean Squared Error (MSE)** loss function.
- The model is trained for 150 epochs with a batch size of 32.
- During training, the model learns the relationships between the sequences of stock prices and uses this information to make predictions on unseen data.

### 4. Making Predictions
- The test set is prepared by creating sequences of 60 previous stock prices from the combined dataset (including both training and testing data).
- The trained model predicts the stock prices for the test set based on these sequences.
- The predicted stock prices are then **denormalized** back to their original scale using the inverse transform of the `MinMaxScaler`.

### 5. Visualization of Results
- The real Google stock prices from the test set are compared to the predicted stock prices.
- **Matplotlib** is used to visualize the comparison between real and predicted stock prices.

## Output
<p align="center">
  <img src="https://github.com/user-attachments/assets/24c62174-5479-4c3e-961d-59699179dc84" width="45%" />
</p>
