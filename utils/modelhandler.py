#Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import time
import pandas as pd

#The Modelhandler class contains usefull methods to compile, fit, evaluate, and plot models
class Modelhandler():

    #This method plots 1. training and validation loss & 2. prediction results
    def plot_model_predictions(self, model, history, y_test, X_test, batch_size, plt_length=200):
        # Plot training and validation loss
        plt.figure(figsize=(15, 3))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions on the test set
        y_pred = model.predict(X_test, batch_size=batch_size)

        # Plot prediction results
        plt.figure(figsize=(10, 3))
        plt.plot(y_test[:plt_length], label='True')
        plt.plot(y_pred[:plt_length], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.legend()
        plt.show()

    #This method compiles the model using Adam optimizer, fits the model, and evaluates it
    def compile_fit_evaluate_model(self, model, loss, metrics, X_train, y_train, max_epochs, batch_size, X_val, y_val, X_test, y_test, callbacks):
        #Compile the model
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=metrics)
        # Train the model
        history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1,)
        #Evaluate the model on test dataset
        test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        print("Loss: ", test_loss)
    
        df_results = callbacks[1].get_training_times_df()
        df_results.set_index('Epoch', inplace=True)
        df_results['train_mse'] = history.history['loss']
        df_results['train_rmse'] = history.history['root_mean_squared_error']
        df_results['train_mape'] = history.history['mean_absolute_percentage_error']
        df_results['train_mae'] = history.history['mean_absolute_error']
        df_results['val_mse'] = history.history['val_loss']
        df_results['val_rmse'] = history.history['val_root_mean_squared_error']
        df_results['val_mape'] = history.history['val_mean_absolute_percentage_error']
        df_results['val_mae'] = history.history['val_mean_absolute_error']
        df_results['test_mse'] = test_loss[0]
        df_results['test_rmse'] = test_loss[1]
        df_results['test_mape'] = test_loss[2]
        df_results['test_mae'] = test_loss[3]

        
        summary_row = {
            'avg_num_eppochs': df_results.index[-1], 
            'avg_train_time_epoch': df_results['Epoch Train_time'].mean(), 
            'avg_total_train_time': df_results['Total Training Time'].mean(), 
            'avg_test_rmse': df_results['test_rmse'].min(), 
            'avg_test_mae': df_results['test_mae'].min()
        }
        df_summary = pd.DataFrame(columns=['avg_num_eppochs', 'avg_train_time_epoch', 'avg_total_train_time', 'avg_test_rmse', 'avg_test_mae'])
        df_summary.loc[len(df_summary)] = summary_row
        df_summary

        return history, df_results, df_summary

    #This methods fits, predicts, and plots the results for sklearn models
    def statistical_model_compile_fit_evaluate(self, X_train, y_train, X_test, y_test, model):
        X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        model.fit(X_train_flattened, y_train)

        X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        y_pred = model.predict(X_test_flattened)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Plot the actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.title('Model Prediction vs Actual')
        plt.legend()
        plt.show()

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