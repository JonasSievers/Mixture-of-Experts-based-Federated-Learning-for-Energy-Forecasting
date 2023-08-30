#Imports

#Plotting
import matplotlib.pyplot as plt
import tensorflow as tf

#For svm model
from sklearn.metrics import mean_squared_error



class Modelhandler():

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
        plt.ylabel('User')
        plt.legend()
        plt.show()

    def compile_fit_evaluate_model(self, model, loss, metrics, X_train, y_train, max_epochs, batch_size, X_val, y_val, X_test, y_test, callbacks):
        
        #Compile the model
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=metrics)
        # Train the model
        history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)
        #Evaluate the model on test dataset
        test_loss = model.evaluate(X_test, y_test, batch_size=batch_size)
        print("Loss: ", test_loss)

        return history

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
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Model Prediction vs Actual')
        plt.legend()
        plt.show()