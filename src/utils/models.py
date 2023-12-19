#Imports
#Imports
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from keras import layers, models


# Custom loss function for MoE model
def custom_mse_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Additional loss function for importance regularization
class ImportanceRegularizationLayer(layers.Layer):
    def __init__(
            self, w_importance=0.001, min_importance=0.001, l2_weight=0, 
            dynamic_reg_strength=0.7, ortho_weight=0.001, sparse_weight=0, **kwargs
            ):
        super(ImportanceRegularizationLayer, self).__init__(**kwargs)
        self.w_importance = w_importance
        self.min_importance = min_importance
        self.l2_weight = l2_weight
        self.dynamic_reg_strength = dynamic_reg_strength
        self.ortho_weight = ortho_weight
        self.sparse_weight = sparse_weight

    def call(self, routing_logits, expert_outputs, num_experts=4):
        
        # Balanced expert utilization
        # Calculate the importance of each expert relative to a batch of training examples
        expert_importance = tf.reduce_sum(routing_logits, axis=0)  # Batchwise sum of gate values for each expert
        # Calculate the coefficient of variation
        cv = tf.math.reduce_std(expert_importance) / tf.math.reduce_mean(expert_importance)
        # Importance loss is the square of the coefficient of variation multiplied by the scaling factor
        cv_loss = self.w_importance * tf.square(cv)
        # Penalty for importance values close to 0
        min_importance_penalty = tf.reduce_sum(tf.nn.relu(self.min_importance - expert_importance))

        # L2 regularization term
        if self.l2_weight != 0:
            l2_loss = self.l2_weight * tf.reduce_sum(tf.square(expert_importance))
        else: 
            l2_loss = 0
        
        # Normalize expert outputs
        reshaped_expert_outputs = tf.reshape(expert_outputs, [16 * 24, num_experts * 8])
        expert_outputs_norm = tf.nn.l2_normalize(reshaped_expert_outputs, axis=-1)
        # Compute the inner product of normalized expert outputs
        outputs_inner_product = tf.matmul(expert_outputs_norm, expert_outputs_norm, transpose_b=True)
        # Penalize similarity of outputs
        outputs_ortho_loss = self.ortho_weight * tf.reduce_sum(tf.square(outputs_inner_product - tf.eye(tf.shape(outputs_inner_product)[0])))

        # Normalize expert weights
        expert_weights_norm = tf.nn.l2_normalize(expert_outputs, axis=0)
        # Compute the inner product of normalized expert weights
        weights_inner_product = tf.matmul(expert_weights_norm, expert_weights_norm, transpose_a=True)
        # Penalize similarity of weights
        weights_ortho_loss = self.ortho_weight * tf.reduce_sum(tf.square(weights_inner_product))

        # Combine the penalties
        combined_ortho_loss = outputs_ortho_loss + weights_ortho_loss  # Or use a different combination strategy

        # Sparsity regularization term
        sparse_loss = self.sparse_weight * tf.reduce_sum(tf.abs(expert_importance))

        # Total loss is the sum of individual losses
        loss = cv_loss + min_importance_penalty + l2_loss + combined_ortho_loss + sparse_loss

        # Add the importance loss to the model's loss
        self.add_loss(loss, inputs=routing_logits)

        # Update dynamic regularization strength for cv_loss
        self.w_importance *= self.dynamic_reg_strength
        self.ortho_weight *= self.dynamic_reg_strength

        return routing_logits

class EinsumLayer(tf.keras.layers.Layer):
    def __init__(self, equation, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation

    def call(self, inputs):
        return tf.einsum(self.equation, *inputs)

class TopKLayer(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        return tf.math.top_k(inputs, k=self.k)
    
#Builds the expert models for the MoE Layer
def build_expert_network(expert_units):
      expert = tf.keras.Sequential([
              layers.Dense(expert_units, activation="relu"), 
              ])
      return expert
   
#Builds a MoE model with soft gating
def build_soft_dense_moe_model(X_train, batch_size, horizon, dense_units,  expert_units, num_experts, metrics):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)

    #experts
    experts = [build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    # Einsum Layer in extra class to enable serialization
    einsum_layer = EinsumLayer('bsn,bnse->bse')
    moe_output = einsum_layer([routing_logits, expert_outputs])
    #END MOE LAYER

    # Add ImportanceRegularizationLayer to the model
    importance_regularizer = ImportanceRegularizationLayer()
    routing_logits = importance_regularizer(routing_logits, expert_outputs)

    x = layers.Dense(dense_units, activation="relu")(moe_output)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)

    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_dense_moe")
    softgated_moe_model.compile(loss=custom_mse_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    
    return softgated_moe_model

#Builds a MoE model with soft gating
def build_soft_biLSTM_moe_model(X_train, batch_size, horizon, lstm_units, num_experts, expert_units, metrics):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)

    #experts
    experts = [build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    # Einsum Layer in extra class to enable serialization
    einsum_layer = EinsumLayer('bsn,bnse->bse')
    moe_output = einsum_layer([routing_logits, expert_outputs])
    #END MOE LAYER

    # Add ImportanceRegularizationLayer to the model
    importance_regularizer = ImportanceRegularizationLayer()
    routing_logits = importance_regularizer(routing_logits, expert_outputs)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(moe_output)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)

    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_lstm_moe")
    softgated_moe_model.compile(loss=custom_mse_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    
    return softgated_moe_model
    

#Builds a MoE model with top_k gating
def build_topk_dense_moe_model(X_train, batch_size, horizon, dense_units, num_experts, top_k, expert_units, metrics):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    router_inputs = x
    router_probs = layers.Dense(num_experts, activation='softmax')(router_inputs)

    # Use custom TopKLayer
    top_k_layer = TopKLayer(top_k)
    expert_gate, expert_index = top_k_layer(router_probs)
        
    expert_idx_mask = tf.one_hot(expert_index, depth=num_experts)
    
    # Use EinsumLayer for einsum operations
    combined_tensor = EinsumLayer('abc,abcd->abd')([expert_gate, expert_idx_mask])
    expert_inputs = EinsumLayer('abc,abd->dabc')([router_inputs, combined_tensor])

    expert_input_list = tf.unstack(expert_inputs, axis=0)
    expert_output_list = [
            [build_expert_network(expert_units=expert_units) for _ in range(num_experts)][idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
    expert_outputs = tf.stack(expert_output_list, axis=1)
    expert_outputs_combined = EinsumLayer('abcd,ace->acd')([expert_outputs, combined_tensor])
    moe_output = expert_outputs_combined

    # Add ImportanceRegularizationLayer
    importance_layer = ImportanceRegularizationLayer()
    _ = importance_layer(router_probs, expert_outputs, num_experts)

    #BOTTOM Model
    x = layers.Dense(dense_units)(moe_output) 
    x = layers.Dense(dense_units, activation="relu")(x)
    
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    topk_moe_model = models.Model(inputs=inputs, outputs=outputs, name="topk_dense_moe")
    topk_moe_model.compile(loss=custom_mse_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)

    return topk_moe_model


#Builds a MoE model with top_k gating
def build_topk_bilstm_moe_model(X_train, batch_size, horizon, lstm_units, num_experts, top_k, expert_units, metrics):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    router_inputs = x
    router_probs = layers.Dense(num_experts, activation='softmax')(router_inputs)

    # Use custom TopKLayer
    top_k_layer = TopKLayer(top_k)
    expert_gate, expert_index = top_k_layer(router_probs)
        
    expert_idx_mask = tf.one_hot(expert_index, depth=num_experts)
    
    # Use EinsumLayer for einsum operations
    combined_tensor = EinsumLayer('abc,abcd->abd')([expert_gate, expert_idx_mask])
    expert_inputs = EinsumLayer('abc,abd->dabc')([router_inputs, combined_tensor])
    
    expert_input_list = tf.unstack(expert_inputs, axis=0)
    expert_output_list = [
            [build_expert_network(expert_units=expert_units) for _ in range(num_experts)][idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
    expert_outputs = tf.stack(expert_output_list, axis=1)
    expert_outputs_combined = EinsumLayer('abcd,ace->acd')([expert_outputs, combined_tensor])
    moe_output = expert_outputs_combined

    # Add ImportanceRegularizationLayer
    importance_layer = ImportanceRegularizationLayer()
    _ = importance_layer(router_probs, expert_outputs, num_experts)

    #BOTTOM Model
    x = layers.Bidirectional(layers.LSTM(lstm_units, activation="relu", return_sequences=True))(moe_output)   
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)

    topk_moe_model = models.Model(inputs=inputs, outputs=outputs, name="topk_lstm_moe")
    topk_moe_model.compile(loss=custom_mse_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)

    return topk_moe_model

def build_bilstm_model(X_train, horizon, num_layers, units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 
    x =  layers.Bidirectional(layers.LSTM(units, return_sequences=True))(input_data)
    for _ in range(num_layers-1):
      x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(horizon)(x) 

    bilstm_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")

    return bilstm_model

def build_cnn_model(X_train, horizon, num_layers, filter, kernel_size, dense_units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 

    x =  layers.Conv1D(filters=filter, kernel_size=kernel_size)(input_data)
    for _ in range(num_layers-1):
      x = layers.Conv1D(filters=filter, kernel_size=kernel_size)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(dense_units)(x)

    output = layers.Dense(horizon)(x) 

    cnn_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")

    return cnn_model

def encoder(x, num_heads, num_features):

    ec_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(x, x)
    ec_norm = layers.LayerNormalization(epsilon=1e-6)(x + ec_att)
    ec_ffn = layers.Dense(num_features, activation='relu')(ec_norm) 
    ec_drop = layers.Dropout(0.2)(ec_ffn) 
    ec_out = layers.LayerNormalization(epsilon=1e-6)(ec_norm + ec_drop)

    return ec_out
  
def decoder(input_data, x, num_heads, num_features):
    dc_att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(input_data, input_data)
    dc_norm1 = layers.LayerNormalization(epsilon=1e-6)(input_data + dc_att1)

    dc_att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(dc_norm1, x)
    dc_norm2 = layers.LayerNormalization(epsilon=1e-6)(dc_norm1 + dc_att2)
    dc_ffn = layers.Dense(num_features, activation='relu')(dc_norm2) 
    dc_drop = layers.Dropout(0.2)(dc_ffn) 
    dc_out = layers.LayerNormalization(epsilon=1e-6)(dc_norm2 + dc_drop)
    return dc_out

def build_transformer_model(X_train, horizon, batch_size, sequence_length, num_layers, num_features, num_heads, dense_units):
    
    #Input Layer
    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size)  
    positional_encoding = layers.Embedding(input_dim=sequence_length-1, output_dim=num_features)(tf.range(sequence_length-1))
    input = input_data + positional_encoding

    #Encoder
    x = encoder(input, num_heads, num_features)
    for _ in range (num_layers-1): 
      x = encoder(x, num_heads, num_features)

    #Decoder
    x = decoder(input, x, num_heads, num_features)
    for _ in range (num_layers-1): 
      x = decoder(input, x, num_heads, num_features)

    # Global average pooling
    output = tf.keras.layers.GlobalAveragePooling1D()(x)  
    output = layers.Dense(dense_units)(output) 
    output = layers.Dense(horizon)(output)

    transformer_model = tf.keras.Model(inputs=input_data, outputs=output)

    return transformer_model