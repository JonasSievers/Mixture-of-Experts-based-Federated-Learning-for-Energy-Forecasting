#Normalization
from sklearn.preprocessing import MinMaxScaler
#Data handling
import pandas as pd
#Create data arrays
import numpy as np

class Datahandler(): 

    #Min Max Scaling
    def min_max_scaling(self, df): #normailizing
        #Min Max Sclaing
        col_names = df.columns
        features = df[col_names]
        scaler = MinMaxScaler().fit(features.values)
        features = scaler.transform(features.values)
        df_scaled = pd.DataFrame(features, columns = col_names, index=df.index)
        return df_scaled

    def standardizing_df(df):
        #Normalize Data
        mean = df.mean()
        std = df.std()

        normalized_df = (df - mean) / std
        return normalized_df

    #Split the datasets into sequences of lngth=Sequence_length
    def create_sequences(self, df, sequence_length):
        sequences = []
        for i in range(len(df) - sequence_length + 1):
            sequence = df.iloc[i:i+sequence_length, :]  # Take all columns
            sequences.append(sequence.values)
        return np.array(sequences)


    # Split each sequence into X (features) and Y (labels)
    def prepare_data(self, sequences, batch_size):

        X = sequences[:, :-1, :].astype('float32') #For all sequences, Exclude last row of the sequence, take all columns
        y = sequences[:, -1, 0].astype('float32') #For all sequences, Take the last row of the sequence, take the first column

        #Adjust the training dataset size to be divisible by batch_size 
        # by discarding the remaining data points that don't fit into a complete batch.
        num_batches = len(X) // batch_size
        # Adjust the training dataset to contain only complete batches
        adjusted_X = X[:num_batches * batch_size]
        adjusted_y = y[:num_batches * batch_size]

        return adjusted_X, adjusted_y
    