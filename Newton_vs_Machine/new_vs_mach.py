import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# Setting a seed for reproducibility so I can ensure consistent results for different runs 
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Loading the training data from CSV files for T <=3.9
# The input data (train_X) and target values (train_Y) are being read into NumPy arrays
# xhecking the shape of the data to check if its properly loaded and the dimensions match
print("Loading data...")
train_X = pd.read_csv("train_X.csv").values # feature data
train_Y = pd.read_csv("train_Y.csv").values # target data
print(f"Input shape: {train_X.shape}, Target shape: {train_Y.shape}")

# Setting up the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(train_X.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(train_Y.shape[1], activation='linear')  
# ])

# Initializing the model as a sequential neural network
# Sequential API - to create a feedforward neural network
model = tf.keras.Sequential()

# Adding the Input layer - first of the Dense Input layer
#                        - with 128 neurons and ReLU activation
# Defining the input shape based on the number of features in train_X
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(train_X.shape[1],)))

# Adding 9 hidden layers, each with 128 neurons and ReLU activation (10 including the input layer)
# Looping to create a deep network with uniform architecture across layers
for _ in range(9):
    model.add(tf.keras.layers.Dense(128, activation='relu'))

# Adding the output layer -  with the same number of neurons as target variables
# For this I am using a linear activation function for regression tasks
model.add(tf.keras.layers.Dense(train_Y.shape[1], activation='linear'))
# model.add(tf.keras.layers.Dense(train_Y.shape[1], activation='relu'))

# Compiling the model
# I initially used mse as my loss function which seemed to work better if predictions 
# require higher precision and is commonly used for physics datasets
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Compiling the model with the Adam optimizer and Mean Absolute Error (MAE) loss, to adhere to simulating the paper
# Configuring the optimizer and metrics for training
# Using MAE as both the loss and a metric for performance evaluation
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
print("Model summary:")
model.summary()


# Training the model and defining the training parameters according to the paper and assignment requirements

# epochs - number of times the model will iterate over the training data
epochs = 1000
#batch_size - number of samples per gradient update
batch_size = 5000
# validation_split - fraction of training data reserved for validation 
validation_split = 0.01  
# validation_split = 0.1  

# Here, I am not sure why if we were to do manual splitting, we should use Spark instead, I used validation_split 
# but thats a doubtt I need to address


# Printing the training start message with configuration details
print(f"Training for {epochs} epochs with batch size {batch_size}...")

# Starting a timer to keep checking training duration
start_time = time.time()

# Training the model using the fit method
# Splitting the training data into training and validation subsets automatically
# Logging the loss and metrics at each epoch for both subsets
history = model.fit(
    train_X, train_Y,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    verbose=1
)

# Stopping the timer to record the total training time
end_time = time.time()
print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")

# Extracting the training history to plot performance metrics
# Plotting the Mean Absolute Error (MAE) for training and validation sets across epochs

history_dict = history.history            # Retrieving the history dictionary containing training and validation metrics
plt.figure(figsize=(10, 6))  
plt.plot(history_dict['mae'], label='Training MAE')         # Plotting training MAE
plt.plot(history_dict['val_mae'], label='Validation MAE')   # Plotting validation MAE
plt.xlabel('Epochs')  
plt.ylabel('Mean Absolute Error')  
plt.yscale('log')                         # Using a logarithmic scale for better visualization of MAE
plt.title('MAE vs Epoch')  
plt.legend()  
plt.grid(True)  
plt.savefig('error_graph1.png')  
print("MAE graph saved as 'error_graph1.png'")  
plt.close()  

# Evaluating the model on the test data
# Using the validation split portion as the test set
# final_loss, final_mae = model.evaluate(
#     train_X[-int(len(train_X) * validation_split):],
#     train_Y[-int(len(train_Y) * validation_split):],
#     verbose=0
# )

# # Print the final test accuracy (MAE)
# print(f"Final Test Loss (MAE): {final_loss:.4f}")
# print(f"Final Test Accuracy (MAE): {final_mae:.4f}")

# this above method did not work for some reason, and ended up givinf same values for loss and accuracy
# so tried another way which worked


# Calculating final training and validation accuracies as 1 - MAE
# Accuracy is being interpreted as the inverse of the MAE for this regression task
final_train_acc = 1 - history.history['mae'][-1]
final_val_acc = 1 - history.history['val_mae'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Saving the trained model to an HDF5 file
# Storing the model for future use if needed - this includes weights and architecture
model.save("newton_vs_machine_model.h5")
print("Model saved as newton_vs_machine_model.h5")