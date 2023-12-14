import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import numpy as np
import keras
import argparse
import os
import sys
import json

def load_data(input_file):
    """
    Load training data from disk
    :param input_file: path to input file
    :return: NxF features matrix (state), and Nx2 target values (action)
    """

    target_frame = None
    head_x = []
    head_y = []
    head_z = []
    joint_1 = []
    joint_2 = []
    joint_3 = []
    joint_4 = []
    action_1 = []
    action_2 = []
    action_3 = []
    action_4 = []
    
    with open(input_file, 'r') as file:
            # Load the JSON data from the file
            data_list = [json.loads(line) for line in file]
            # Iterate through the data
            for entry in data_list:
                assert len(entry) == 12, "Expected each line to have 12 elements."

                head_x.append(float(entry[ "head_position_x"]))
                head_y.append(float(entry[ "head_position_y"]))
                head_z.append(float(entry[ "head_position_z"]))
                joint_1.append(float(entry[ "joint_1"]))
                joint_2.append(float(entry[ "joint_2"]))
                joint_3.append(float(entry[ "joint_3"]))
                joint_4.append(float(entry[ "joint_4"]))
                action_1.append(float(entry[ "next_joint_1"]))
                action_2.append(float(entry[ "next_joint_2"]))
                action_3.append(float(entry[ "next_joint_3"]))
                action_4.append(float(entry[ "next_joint_4"]))

    features = np.column_stack((head_x, head_y, head_z, joint_1, joint_2, joint_3, joint_4))
    targets = np.column_stack((action_1, action_2, action_3, action_4))

    return features, targets

def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev


def normalize_data_per_row(data, mean, stdev):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :param mean: mean for normalization
    :param stdev: standard deviation for normalization
    :return: whitened data, (data - mean) / stdev
    """

    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    centered = data - np.tile(mean, (data.shape[0], 1))
    normalized_data = np.divide(centered, np.tile(stdev, (data.shape[0],1)))

    return normalized_data

def build_model(input_shape):
    # Build a simple neural network model
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4)  # Output layer with 2 units for action_1 and action_3
    ])
    # Validation Loss: [0.017627805471420288, 0.07829522341489792]
    
    """
    # LSTM model
    model = keras.Sequential([
         keras.layers.LSTM(50, activation='relu', input_shape=(1, 7)),
         keras.layers.Dense(64, activation='softmax'),
         keras.layers.Dense(4)
    ])
    # Validation Loss: [0.018657149747014046, 0.08205969631671906]
    """
    """
    # LSTM model2: Best model so far
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1, 7)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.LSTM(32, activation='relu'),
        keras.layers.Dense(4)
    ])
    # Validation Loss: [0.017765656113624573, 0.07670820504426956]
    """
    return model

    


    # model = keras.Sequential([
    #     keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(2)  # Output layer with 2 units for action_1 and action_3
    # ])
    # return model



def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.001, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # normalize
    #norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    #norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss='mse',
                 metrics=['mae'])
        
    # tensorboard callback
    logs_dir = '/Users/tetsu/Documents/School/Class/CPSC459/bim-project/imitation_learning_logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'imitation_learning_weights.h5'),
                                                            monitor='mae',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)
    
    # do training for the specified number of epochs and with the given batch size
    model.fit(train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(val_input, val_target),
             callbacks=[tbCallBack, checkpointCallBack])
    
    # Save model
    
    model.save("/Users/tetsu/Documents/School/Class/CPSC459/bim-project/imitation_learning_models/model", overwrite=True, save_format = 'h5')

def validate_model(model, features_val, targets_val):
    validation_loss = model.evaluate(features_val, targets_val)
    print(f"Validation Loss: {validation_loss}")
    

def predict(model, features_val, targets_val):
    # Make predictions on the validation data
    predictions = model.predict(features_val)

    
    # Plot and compare predictions with true targets
    plt.scatter(targets_val[:, 0], targets_val[:, 1],label='True Actions')
    plt.scatter(predictions[:, 0], predictions[:, 1], label='Predicted Actions')
    
    plt.legend()
    plt.xlabel('Action 1')
    plt.ylabel('Action 2')
    plt.show()
    
    plt.scatter(targets_val[:, 2], targets_val[:, 3], label='True Actions')
    plt.scatter(predictions[:, 2], predictions[:, 3], label='Predicted Actions')
    
    plt.legend()
    plt.xlabel('Action 3')
    plt.ylabel('Action 4')
    plt.show()

if __name__ == "__main__":

    # Handling command line arguments
    file_path = "/Users/tetsu/Documents/School/Class/CPSC459/bim-project/output.json"

    # Check file exists and load data
    if os.path.exists(file_path):
        features, targets = load_data(file_path)
    else:
        print(f"The file at {file_path} does not exist.")
        sys.exit(1)

    # Split data into training and validation sets
    features_train, features_val, targets_train, targets_val = train_test_split(
        features, targets, test_size=0.2, random_state=42)

    # features_train = np.reshape(features_train, (features_train.shape[0], 1, features_train.shape[1]))
    # features_val = np.reshape(features_val, (features_val.shape[0], 1, features_val.shape[1]))
    # Build model
    model = build_model(features_train.shape[1])

    # Compute normalization parameters
    input_mean, input_stdev = compute_normalization_parameters(features)

    # Train model
    train_model(model, features_train, targets_train, features_val, targets_val, 
                input_mean, input_stdev, epochs = 1000, learning_rate = 0.0001, batch_size = 100)

    # Validate model
    validate_model(model, features_val, targets_val)

    # Use model to predict and plot
    predict(model, features_val, targets_val)