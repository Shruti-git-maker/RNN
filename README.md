# RNN
Overview
This notebook demonstrates the implementation of Recurrent Neural Networks (RNNs) using both TensorFlow and PyTorch. It covers creating a simple RNN model in TensorFlow and a more detailed PyTorch implementation, including data preparation, model training, and prediction visualization.

Features
1. TensorFlow RNN Model
Model Architecture:

Embedding layer for input dimensionality reduction.

LSTM (Long Short-Term Memory) layer for sequence processing.

Dense layer for output classification.

Model Summary:

Displays the architecture and parameter count of the model.

2. PyTorch RNN Model
Synthetic Dataset Generation:

Creates a dataset of sine waves for time series forecasting.

Data Preparation:

Converts data into PyTorch tensors.

Reshapes data for RNN input format.

Model Definition:

Custom RNN class using PyTorch's nn.Module.

Includes an RNN layer followed by a fully connected layer.

Training Loop:

Trains the model using Mean Squared Error (MSE) loss and Adam optimizer.

Plots training loss over epochs.

Prediction and Visualization:

Makes predictions on the training data.

Plots true vs. predicted values for comparison.

Dependencies
The following Python libraries are required:

tensorflow: For building the TensorFlow RNN model.

torch: For building the PyTorch RNN model.

numpy: For numerical computations.

matplotlib: For plotting visualizations.

Installation
To install the required libraries, run:

bash
pip install tensorflow torch numpy matplotlib
Usage Instructions
Clone or download the notebook file to your local machine.

Open the notebook in Jupyter Notebook or Google Colab.

Run all cells sequentially to execute the analysis.

Key Sections
TensorFlow Model
Model Architecture:

python
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=64),
    layers.LSTM(128),
    layers.Dense(10)
])
Model Summary:

Displays the architecture and parameter count.

PyTorch Model
Synthetic Dataset Generation:

Creates sine wave data for time series forecasting.

Example:

python
X, y = [], []
for i in range(1000):
    x = np.linspace(i * np.pi * 2, (i+1) * np.pi * 2, 51)
    sine_wave = np.sin(x)
    X.append(sine_wave[:-1])
    y.append(sine_wave[1:])
Data Preparation:

Converts data to PyTorch tensors and reshapes for RNN input.

Example:

python
X = torch.tensor(X).float()
y = torch.tensor(y).float()
X_train = X.unsqueeze(2)
y_train = y.unsqueeze(2)
Model Definition:

Custom RNN class with an RNN layer followed by a fully connected layer.

Example:

python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
Training Loop:

Trains the model using MSE loss and Adam optimizer.

Example:

python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
Prediction and Visualization:

Makes predictions on the training data.

Plots true vs. predicted values.

Example:

python
with torch.no_grad():
    predictions = model(X_train)
plt.plot(y_train.numpy()[0], label='True')
plt.plot(predictions.numpy()[0], label='Predicted')
plt.legend()
plt.show()
Observations
The TensorFlow model provides a simple RNN architecture for sequence processing.

The PyTorch model demonstrates a more detailed implementation with custom RNN class and training loop.

The PyTorch model effectively predicts sine wave values, showing its capability in time series forecasting.

Future Improvements
Implement more advanced RNN architectures like GRU or LSTM in PyTorch.

Use real-world datasets for more practical applications.

Explore additional techniques like data augmentation or ensemble methods to improve model performance.

License
This project is open-source and available under the MIT License.
