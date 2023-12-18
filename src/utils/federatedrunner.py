import wandb
from keras import backend as K
from utils.models import *
from utils.datahandling import *

def avg_weights_with_noise_fedprox(weight_list, clip_threshold=None, noise_scale=0.001, proximal_term=0.1):
    avg_grad = list()

    for grad_list_tuple in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)

        if clip_threshold is not None:
            layer_mean = tf.clip_by_value(layer_mean, -clip_threshold, clip_threshold)

        noise = tf.random.normal(shape=layer_mean.shape, mean=0.0, stddev=noise_scale)
        noisy_layer_mean = layer_mean + noise

        # Add FedProx proximal term
        proximal_update = -proximal_term * noisy_layer_mean

        avg_grad.append(noisy_layer_mean + proximal_update)

    return avg_grad

def run_federated_benchmark_model(
    num_clusters,
    federated_rounds,
    cluster_users,
    save_path,
    wb_project_name,
    wb_project,
    wb_model_name,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon,
    loss,
    metrics,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    layers = "",
    lstm_units = "",
    cnn_filter = "",
    cnn_kernel_size = "",
    cnn_dense_units = "",
    transformer_dense_units = "",
    sequence_length = "",
    transformer_num_features = "",
    transformer_num_heads = ""
):
    # Create global models for each cluser (10)
    for cluster in range(num_clusters):

        #Build and save global model
        if 'lstm' in wb_model_name.lower():
            global_model = build_bilstm_model(
                X_train[f'user{user_index}'], 
                horizon, 
                num_layers=layers, 
                units=lstm_units, 
                batch_size=batch_size)
        elif 'cnn' in wb_model_name.lower():
            global_model = build_cnn_model(
                X_train[f'user{user_index}'], 
                horizon, 
                num_layers=layers,
                filter=cnn_filter,
                kernel_size=cnn_kernel_size, 
                dense_units=cnn_dense_units, 
                batch_size=batch_size)
        elif 'tran' in wb_model_name.lower():
            global_model = build_transformer_model(
                X_train[f'user{user_index}'], 
                horizon, 
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_layers=layers,
                num_features = transformer_num_features,
                num_heads=transformer_num_heads,
                dense_units=transformer_dense_units, 
                )
        global_model.save(f'{save_path}/wandb/{wb_project}_{wb_model_name}_c{cluster}_FLround{0}.keras')

    for federated_round  in range(federated_rounds):
        print("Started Federated training round ----------", federated_round+1, f"/ {federated_rounds}")
        for cluster_number, users_in_cluster in cluster_users.items():
            print(f"Cluster {cluster_number}:")

            #Get global models weights
            global_model = tf.keras.models.load_model(f'{save_path}/wandb/{wb_project}_{wb_model_name}_c{cluster_number}.keras', compile=False)
            global_model_weights = global_model.get_weights()

            #initial list for local model weights
            local_model_weight_list = list()

            #for idx, user in enumerate(df_array): 
            for user_index in users_in_cluster:
                user_df = df_array[user_index-1]  # Get the user's DataFrame from the array
                print(f"User {user_index}") 

                # Initialize wandb
                wandb.init(project=wb_project_name, 
                    name=f'{wb_model_name}_u{user_index}_FLrd{federated_round+1}',
                    config={
                    'max_epochs': max_epochs,
                    'batch_size': batch_size,
                    'optimizer': "Adam",
                    'learning_rate': 0.001,
                    'architecture': wb_model_name,
                })

                #build and compile local model X_train, batch_size, horizon, dense_units,  expert_units, num_experts, m1
                if 'lstm' in wb_model_name.lower():
                    local_model = build_bilstm_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        num_layers=layers, 
                        units=lstm_units, 
                        batch_size=batch_size)
                elif 'cnn' in wb_model_name.lower():
                    local_model = build_cnn_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        num_layers=layers,
                        filter=cnn_filter,
                        kernel_size=cnn_kernel_size, 
                        dense_units=cnn_dense_units, 
                        batch_size=batch_size)
                elif 'tran' in wb_model_name.lower():
                    local_model = build_transformer_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        num_layers=layers,
                        num_features = transformer_num_features,
                        num_heads=transformer_num_heads,
                        dense_units=transformer_dense_units, 
                        )
                local_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006), metrics=metrics)

                #set local model weight to the weight of the global model
                local_model.set_weights(global_model_weights)
                #Fit local model to local data
                histroy, user_results = fit_evaluate_model(
                    model=local_model, 
                    X_train=X_train[f'user{user_index}'],
                    y_train = y_train[f'user{user_index}'], 
                    max_epochs = max_epochs, 
                    batch_size=batch_size, 
                    X_val=X_val[f'user{user_index}'], 
                    y_val=y_val[f'user{user_index}'], 
                    X_test=X_test[f'user{user_index}'], 
                    y_test=y_test[f'user{user_index}'], 
                    callbacks=callbacks, 
                    user=f'user{user_index}', 
                    hyper=wb_model_name,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
                )
                
                #add model weights to list        
                local_model_weights = local_model.get_weights()
                local_model_weight_list.append(local_model_weights)
                            
                #clear session to free memory after each communication round
                wandb.finish()
                K.clear_session()
            
            #to get the average over all the local model, we simply take the sum of the scaled weights
            average_weights = avg_weights_with_noise_fedprox(local_model_weight_list)
            #update global model 
            global_model.set_weights(average_weights)
            #Save global models
            global_model.save(f"{save_path}/wandb/{wb_project}_{wb_model_name}_c{cluster_number}_FLround{federated_round+1}.keras")
            print("Saved Global models")

def evaluate_federated_benchmark_model(
    federated_rounds,
    save_path,
    wb_project,
    wb_model_name,
    cluster_users,
    num_rounds,
    horizon,
    batch_size,
    metrics,
    loss,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    df_array,
    results,
    all_results,
    layers = "",
    lstm_units = "",
    cnn_filter = "",
    cnn_kernel_size = "",
    cnn_dense_units = "",
    transformer_dense_units = "",
    sequence_length = "",
    transformer_num_features = "",
    transformer_num_heads = ""
):
    
    for cluster_number, users_in_cluster in cluster_users.items():
        print(f"Cluster {cluster_number}:")
        for user_index in users_in_cluster:
            print("User: ", user_index)
            for round in range(num_rounds):
                global_model = tf.keras.models.load_model(f"{save_path}/wandb/{wb_project}_{wb_model_name}_c{cluster_number}_FLround{federated_rounds}.keras", compile=False)
                
                if 'lstm' in wb_model_name.lower():
                    local_model = build_bilstm_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        num_layers=layers, 
                        units=lstm_units, 
                        batch_size=batch_size)
                elif 'cnn' in wb_model_name.lower():
                    local_model = build_cnn_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        num_layers=layers,
                        filter=cnn_filter,
                        kernel_size=cnn_kernel_size, 
                        dense_units=cnn_dense_units, 
                        batch_size=batch_size)
                elif 'tran' in wb_model_name.lower():
                    local_model = build_transformer_model(
                        X_train[f'user{user_index}'], 
                        horizon, 
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        num_layers=layers,
                        num_features = transformer_num_features,
                        num_heads=transformer_num_heads,
                        dense_units=transformer_dense_units, 
                        )
                
                local_model.set_weights(global_model.get_weights())
                histroy, user_results = compile_fit_evaluate_model(
                    model=local_model, 
                    loss=loss, 
                    metrics=metrics, 
                    X_train=X_train[f'user{user_index}'],
                    y_train = y_train[f'user{user_index}'], 
                    max_epochs = 1, 
                    batch_size=batch_size, 
                    X_val=X_val[f'user{user_index}'], 
                    y_val=y_val[f'user{user_index}'], 
                    X_test=X_test[f'user{user_index}'], 
                    y_test=y_test[f'user{user_index}'], 
                    callbacks=callbacks, 
                    user=f'user{user_index}', 
                    hyper=wb_model_name,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006)
                )
                # Add the 'architecture' column from dense_user_results to dense_results
                all_results = pd.merge(all_results, user_results, how='outer')            

    for idx in range(len(df_array)):
        new_row = {
            'architecture': wb_model_name,
            'train_time': all_results[all_results["user"]==f"user{idx+1}"]["train_time"].mean(), 
            'avg_time_epoch' : all_results[all_results["user"]==f"user{idx+1}"]["avg_time_epoch"].mean(),
            'mse': all_results[all_results["user"]==f"user{idx+1}"]["mse"].mean(),
            'mse_std' : all_results[all_results["user"]==f"user{idx+1}"]["mse"].std(),
            'rmse': all_results[all_results["user"]==f"user{idx+1}"]["rmse"].mean(),
            'rmse_std' : all_results[all_results["user"]==f"user{idx+1}"]["rmse"].std(),
            'mape': all_results[all_results["user"]==f"user{idx+1}"]["mape"].mean(),
            'mape_std' : all_results[all_results["user"]==f"user{idx+1}"]["mape"].std(),
            'mae': all_results[all_results["user"]==f"user{idx+1}"]["mae"].mean(),
            'mae_std' : all_results[all_results["user"]==f"user{idx+1}"]["mae"].std(),
        }
        results.loc[len(results)] = new_row

    print(results.head(1))
    results.to_csv(f'{save_path}/wandb/{wb_model_name}_results.csv')
    all_results.to_csv(f'{save_path}/wandb/{wb_model_name}_all_results.csv')
