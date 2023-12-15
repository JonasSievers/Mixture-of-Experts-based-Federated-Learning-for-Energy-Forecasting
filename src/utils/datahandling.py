# Imports

#Imports
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import wandb


#min_max_scaling
#Sclaes all columns of the dataframe df to the rang (0,1)
def min_max_scaling(df): #normailizing
    #Min Max Sclaing
    col_names = df.columns
    features = df[col_names]
    scaler = MinMaxScaler().fit(features.values)
    features = scaler.transform(features.values)
    df_scaled = pd.DataFrame(features, columns = col_names, index=df.index)
    return df_scaled

#create_sequences
#Split the dataframe into datasets with sequences of lngth=Sequence_length
def create_sequences(df, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequence = df.iloc[i:i+sequence_length, :]  # Take all columns
        sequences.append(sequence.values)
    return np.array(sequences)

#prepare_data
# Split each sequence into X (features) and Y (labels). 
# The label Y must be the FIRST column! The last batch is discarded, when < batch_size
def prepare_data(sequences, batch_size):
    X = sequences[:, :-1, :].astype('float32') #For all sequences, Exclude last row of the sequence, take all columns
    y = sequences[:, -1, 0].astype('float32') #For all sequences, Take the last row of the sequence, take the first column

    #As some models need to reshape the inputs, the correct batch_size is important
    #Adjust the dataset_size to be divisible by batch_size by discarding the remaining data points not fitting a complete batch.
    num_batches = len(X) // batch_size
    adjusted_X = X[:num_batches * batch_size]
    adjusted_y = y[:num_batches * batch_size]

    return adjusted_X, adjusted_y

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

    def get_training_times_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        data = {
            'Epoch': list(range(1, len(self.epoch_times) + 1)),
            'Epoch Train_time': self.epoch_times,
            'Epoch Avg Train_time': average_epoch_times,
            'Total Training Time': total_training_time
        }
        return pd.DataFrame(data)
    

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        self.losses = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.losses['epoch'].append(epoch)
        self.losses['train_loss'].append(logs['loss'])
        self.losses['val_loss'].append(logs['val_loss'])

    def on_test_end(self, logs=None):
        self.losses['test_loss'].append(logs['loss'])

    def get_loss_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        self.losses['avg_epoch_time'] = average_epoch_times
        self.losses['total_training_time'] = total_training_time
        return pd.DataFrame(self.losses)
    
#This method compiles the model using Adam optimizer, fits the model, and evaluates it
def compile_fit_evaluate_model(model, loss, metrics, X_train, y_train, max_epochs, batch_size, X_val, y_val, X_test, y_test, callbacks, user= "", hyper="", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)):
    
    #Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    wandb_callback = wandb.keras.WandbCallback(save_model=False)
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=max_epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val), 
        callbacks=callbacks + [wandb_callback],
        verbose=0,)
    
    #Evaluate the model on test dataset
    test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    train_times = callbacks[1].get_training_times_df()
    total_train_time = train_times["Total Training Time"][0]
    avg_time_epoch = train_times["Epoch Avg Train_time"].iloc[-1]

    model_user_result = pd.DataFrame(
        data=[[user, hyper, total_train_time, avg_time_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]]], 
        columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mape", "mae"]
    )

    wandb.log({
        'Final MSE': test_loss[0],
        'Final RMSE': test_loss[1],
        'Final MAPE': test_loss[2],
        'Final MAE': test_loss[3],
        'Total Training Time': total_train_time,
        'Average Time per Epoch': avg_time_epoch
    })

    return history, model_user_result