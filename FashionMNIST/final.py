# Importing the necessary libraries for this project
# I am using TensorFlow for building the model, Matplotlib for visualization, and Pandas for saving results to a CSV
# This initial code is from class lecture notes where we learnt CNN with dropout
import tensorflow as tf 
# import tensorflow.keras import layers, models
# import tensorflow.keras.datasets import fashion_mnist
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#Loading and preprocessing the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images/255, test_images/255

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # this is the first dropout layer
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # this is the second dropout layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=32, epochs=10, verbose=1, validation_data=(test_images, test_labels))

# trying hyperparameter tuning

import matplotlib.pyplot as plt
import pandas as pd

# I am trying creating and training the model for different parameters 

# Setting up a range of hyperparameters for tuning
# I am experimenting with two dropout rates for two layers, learning rates, and number of epochs
dropout_layer1 = [0.25, 0.3]  
dropout_layer2 = [0.5, 0.6] 
 # Learning rates for the SGD optimizer 
learning_rates = [0.01, 0.005] 
 # Fixed batch size for all experiments
batch_size = 16 
# epochs = 15
# Different numbers of epochs to observe model behavior
epoch_options = [10, 13, 15]  

# Initializing empty lists to track results and training histories
# These lists will store performance metrics and training progress for each hyperparameter combination
results = []
all_histories = []

# Starting the hyperparameter tuning loop
# Iterating through all combinations of the hyperparameter values defined above
for d1 in dropout_layer1:
    for d2 in dropout_layer2:
        for lr in learning_rates:
            for epochs in epoch_options:
                # Printing the current combination of hyperparameters for tracking progress
                print(f"Training with Dropout1={d1}, Dropout2={d2}, Learning Rate={lr}, Epochs={epochs}...")
                
                # Defining the CNN model architecture
                # Using two convolutional layers, max pooling, dropout, and dense layers
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # First conv layer
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(d1), 
                    tf.keras.layers.Flatten(), 
                    tf.keras.layers.Dense(128, activation='relu'),  
                    tf.keras.layers.Dropout(d2),
                    tf.keras.layers.Dense(10, activation='softmax')  
                ])
                
                # Compiling the model with the SGD optimizer
                # Using sparse categorical crossentropy for multi-class classification and accuracy as the evaluation metric
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                
                # Training the model on the training data
                # Also validating the model using the test data during training
                history = model.fit(
                    train_images, train_labels, 
                    batch_size=batch_size,  
                    epochs=epochs,  
                    verbose=1, 
                    validation_data=(test_images, test_labels)  
                )
                
                # Storing the training history for this combination of hyperparameters
                all_histories.append(history)
                
                # Evaluating the model on the test data
                # Capturing the test loss and accuracy
                test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
                print(f"Test Accuracy: {test_accuracy:.4f}")
                
                # Logging the results for this combination of hyperparameters
                # Each result includes the hyperparameter values and the achieved test accuracy
                results.append({
                    'Dropout1': d1,
                    'Dropout2': d2,
                    'Learning Rate': lr,
                    'Batch Size': batch_size,
                    'Epochs': epochs,
                    'Test Accuracy': test_accuracy
                })

# Converting the results into a Pandas DataFrame and saving to a CSV file
# This allows for easy analysis and visualization of the results later
results_df = pd.DataFrame(results)
results_df.to_csv("3hyperparameter_tuning.csv", index=False)
print("Hyperparameter tuning complete. Results saved to 'hyperparameter_tuning_with_epochs.csv'.")

# Plotting the training and validation accuracy for the last training run
# This provides insights into how the model performed over epochs in the final configuration
if all_histories:
    plt.figure(figsize=(8, 6)) 
    last_history = all_histories[-1] 
    plt.plot(last_history.history['accuracy'], label='Training Accuracy')  
    plt.plot(last_history.history['val_accuracy'], label='Validation Accuracy')  
    plt.title('Training and Validation Accuracy for Last Run')  
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  
    plt.legend()  
    
    # Saving the plot as a PNG file for documentation
    plot_path = "3training_validation_accuracy.png"
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
else:
    # Handling the case where no training was completed
    print("No histories to plot. Training may have failed.")