#Imports
#Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import tensorflow_probability as tfp
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#The ModelGenerator class contains methods to build different models
#Tensorflow models: softgated_moe_model, top1_moe_model, topk_moe_model, lstm_model, bilstm_model, cnn, dense, probability_model, transformer
#SKlearn models: Svm, Elasticnet_regression, Decisiontree, Randomforrest, K_neighbors regression
class ModelGenerator():
  
  #Builds the expert models for the MoE Layer
  def build_expert_network(self, ff_dim, embed_dim):
      expert = keras.Sequential([
              layers.Dense(ff_dim, activation="relu"), 
              layers.Dense(embed_dim)
              ])
      return expert

  #Builds a MoE model with soft gating
  def build_softgated_moe_model(self, X_train, batch_size, horizon, dense_1_units, num_experts, expert_hidden_units, expert_output_units, dense_2_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = layers.Dense(dense_1_units, activation="relu", name='dense_layer_1')(inputs)

    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)
    #experts
    experts = [m1.build_expert_network(ff_dim=expert_hidden_units, embed_dim=expert_output_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    moe_output = tf.einsum('bsn,bnse->bse', routing_logits, expert_outputs)
    #END MOE LAYER

    x = layers.Dense(dense_2_units, name='dense_layer_2')(moe_output)
    x = layers.Flatten(name='flat_layer')(x)
    #x = layers.Dense(dense_3_units, name='dense_layer_3')(x)
    outputs = layers.Dense(horizon, name='output_layer')(x)
    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs)

    return softgated_moe_model

  #Builds a MoE model with top_1 gating and expert balancing
  def build_top1_moe_model(self, X_train, batch_size, horizon, sequenze_length, num_experts, expert_capacity, expert_dim, dense_1_units, dense_2_units, dense_3_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = layers.Dense(dense_1_units, activation="relu", name='dense_layer_1')(inputs)

    #EMBEDDED MOE LAYER
    #ROUTER
    router_inputs = tf.reshape(x, [batch_size*sequenze_length, dense_1_units], name="reshape_1")
    router = layers.Dense(num_experts, activation="softmax", name='router_layer')(router_inputs)
    expert_gate, expert_index = tf.math.top_k(router, k=1)
    expert_mask = tf.one_hot(expert_index, depth=num_experts) 
    position_in_expert = tf.cast(tf.math.cumsum(expert_mask, axis=0) * expert_mask, tf.dtypes.int32) 
    expert_mask *= tf.cast(tf.math.less(tf.cast(position_in_expert, tf.dtypes.int32), expert_capacity),tf.dtypes.float32,)
    expert_mask_flat = tf.reduce_sum(expert_mask, axis=-1) 
    expert_gate *= expert_mask_flat 
    combined_tensor = tf.expand_dims(
                expert_gate 
                * expert_mask_flat 
                * tf.squeeze(tf.one_hot(expert_index, depth=num_experts), 1),
                -1,) * tf.squeeze(tf.one_hot(position_in_expert, depth=expert_capacity), 1) 
    dispatch_tensor = tf.cast(combined_tensor, tf.dtypes.float32) 
    #EXPERTS
    experts = [m1.build_expert_network(ff_dim=expert_dim, embed_dim=dense_1_units) for _ in range(num_experts)]
    expert_inputs = tf.einsum("ab,acd->cdb", router_inputs, dispatch_tensor) 
    expert_inputs = tf.reshape(expert_inputs, [num_experts, expert_capacity, dense_1_units], name="reshape_2")
    expert_input_list = tf.unstack(expert_inputs, axis=0) 
    expert_output_list = [
        experts[idx](expert_input)
        for idx, expert_input in enumerate(expert_input_list) 
    ]
    expert_outputs = tf.stack(expert_output_list, axis=1) 
    #Add and Multiply expert models with router probability
    expert_outputs_combined = tf.einsum(
        "abc,xba->xc", expert_outputs, combined_tensor      
    )
    outputs = tf.reshape(expert_outputs_combined, [batch_size, sequenze_length, dense_1_units], name="reshape_3")
    moe_output= layers.Dropout(0.1)(outputs, training=True)
    #END MOE LAYER

    #BOTTOM Model
    x = layers.Dense(dense_2_units, name='dense_layer_2')(moe_output) #(16, 48, 16)
    x = layers.Flatten(name='flat_layer')(x)
    #x = layers.Dense(dense_3_units, name='dense_layer_3')(x)
    outputs = layers.Dense(horizon, name='output_layer')(x)
    moe_model = models.Model(inputs=inputs, outputs=outputs)

    return moe_model

  #Builds a MoE model with top_k gating
  def build_topk_moe_model(self, X_train, batch_size, horizon, dense_1_units, num_experts, top_k, expert_hidden_units, expert_output_units, dense_2_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = layers.Dense(dense_1_units, activation="relu", name='dense_layer_1')(inputs)

    #EMBEDDED MOE LAYER
    # ROUTER
    router_probs = layers.Dense(num_experts, activation='softmax')(x)
    expert_gate, expert_index = tf.math.top_k(router_probs, k=top_k)
    expert_idx_mask = tf.one_hot(expert_index, depth=num_experts)
    combined_tensor = tf.einsum('abc,abcd->abd', expert_gate, expert_idx_mask)
    #EXPERTS
    experts = [m1.build_expert_network(ff_dim=expert_hidden_units, embed_dim=expert_output_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    moe_output = tf.einsum('bsn,bnse->bse', combined_tensor, expert_outputs)
    #END MOE LAYER

    #BOTTOM Model
    x = layers.Dense(dense_2_units, name='dense_layer_2')(moe_output) 
    x = layers.Flatten(name='flat_layer')(x)
    #x = layers.Dense(dense_3_units, name='dense_layer_3')(x)
    outputs = layers.Dense(horizon, name='output_layer')(x)
    topk_moe_model = models.Model(inputs=inputs, outputs=outputs)

    return topk_moe_model
  
  #Builds 
  def build_lstm_model(self, X_train, horizon, lstm_cells):
    #model = tf.keras.Sequential([
    #tf.keras.layers.LSTM(lstm_cells, input_shape=(X_train.shape[1], X_train.shape[2])), 
    #tf.keras.layers.Dense(horizon) 
    #])

    model = tf.keras.Sequential([
        layers.LSTM(lstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True), #shape (48,3)
        layers.LSTM(lstm_cells, return_sequences=True), #If True, retuns all sequences x with (x, 48,1) shape, if false only the (x,1)
        layers.GlobalAveragePooling1D(),
        layers.Dense(horizon) #Output 1 value
    ])

    return model
  

  def build_bilstm_model(self, X_train, horizon, bilstm_cells):
    model = tf.keras.Sequential([
        layers.Bidirectional(tf.keras.layers.LSTM(bilstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)), #shape (48,3)
        layers.Bidirectional(tf.keras.layers.LSTM(bilstm_cells, return_sequences=False)), #If True, retuns all sequences x with (x, 48,1) shape, if false only the (x,1)
        layers.Dropout(0.2),
        layers.Dense(horizon) #Output 1 value
    ])

    return model
  

  def build_cnn_model(self, X_train, horizon, filter, kernel_size):

    model = tf.keras.Sequential([
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      
      layers.GlobalAveragePooling1D(), # tf.keras.layers.Flatten()
      layers.Dense(horizon)
    ])
    return model
  

  def build_dense_model(self, X_train, horizon, units):

    model = tf.keras.Sequential([
        layers.Dense(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.Dense(units, activation='relu'),
        layers.Flatten(),
        layers.Dense(horizon) 
    ])
    return model
  

  def build_probability_model(self, X_train, horizon, units):

    probability_model = tf.keras.Sequential([
      tfp.layers.DenseFlipout(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
      tfp.layers.DenseFlipout(units, activation='relu'),
      tfp.layers.DenseFlipout(horizon)
    ])
    return probability_model

  
  def build_transformer_model(self, X_train, horizon, num_features, num_heads, key_dim):
    
    #Input Layer
    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))  # Replace 'features_dim' with the actual number of features
    input_data = layers.LayerNormalization(epsilon=1e-6)(input_data)
    ###Encoder 
    # Encoder Layer 1
    ec1_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_data, input_data)
    ec1_out1 = layers.LayerNormalization(epsilon=1e-6)(input_data + ec1_attention_output)
    ec1_ffn_output = layers.Dense(64, activation='relu')(ec1_out1) 
    ec1_ffn_output = layers.Dense(num_features, activation='relu')(ec1_ffn_output) 
    ec1_ffn_output = layers.Dropout(0.1)(ec1_ffn_output) 
    ec1_out2 = layers.LayerNormalization(epsilon=1e-6)(ec1_out1 + ec1_ffn_output)

   
    ###Decoder
    # Decoder Layer 1
    dc1_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_data, input_data)
    dc1_out1 = layers.LayerNormalization(epsilon=1e-6)(input_data + dc1_attention_output)

    dc1_cross_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(dc1_out1, ec1_out2)
    dc1_cross_attention_output = layers.Dropout(0.1)(dc1_cross_attention_output)
    dc1_cross_attention_output = layers.LayerNormalization(epsilon=1e-6)(dc1_out1 + dc1_cross_attention_output)

    dc1_ffn_output = layers.Dense(64, activation='relu')(dc1_cross_attention_output)
    dc1_ffn_output = layers.Dense(num_features, activation='relu')(dc1_ffn_output) 
    dc1_ffn_output = layers.Dropout(0.1)(dc1_ffn_output) 
    dc1_decoder_output = layers.LayerNormalization(epsilon=1e-6)(dc1_cross_attention_output + dc1_ffn_output)


    # Global average pooling
    #output = tf.keras.layers.GlobalAveragePooling1D()(out2) 
    #output = tf.keras.layers.LSTM(64)(out2) 
    output = layers.Dense(32)(dc1_decoder_output) 
    output = layers.Dense(horizon)(output)

    model = tf.keras.Model(inputs=input_data, outputs=output)

    return model


  def build_svm_model(self, kernel):
    svm_model = svm.SVR(kernel=kernel)

    return svm_model

  def build_elasticnet_regression_model(self, alpha, l1_ratio):
    elasticnet_regression_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio) #alpha start small, increase when overfitting

    return elasticnet_regression_model

  def build_decisiontree_model():
    decisiontree_model = DecisionTreeRegressor()

    return decisiontree_model

  def build_randomforrest_model(self, n_estimators):
    randomforrest_model = RandomForestRegressor(n_estimators=n_estimators) #, random_state=42

    return randomforrest_model

  def build_k_neighbors_model(self, n_neighbors):
    k_neighbors_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    return k_neighbors_model



