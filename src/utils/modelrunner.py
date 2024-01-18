import wandb
from utils.models import *
from utils.datahandling import *

def run_bilstm_model(
    wb_project_name,
    wb_model_name,
    wb_project,
    save_path,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon, 
    lstm_layers, 
    lstm_units,    
    metrics,
    loss,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    results,
    all_results,
    num_rounds=5
):
    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wb_project_name, 
                name=f'{wb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wb_model_name,
            })
            
            model = build_bilstm_model(X_train[f'user{idx+1}'], horizon, num_layers=lstm_layers, units=lstm_units, batch_size=batch_size)
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer')   

            model.save(f'{save_path}/wandb/{wb_project}_{wb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()
        
    for idx in range(len(df_array)):
        new_row = {
            'architecture': "bilstm",
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


def run_cnn_model(
    wb_project_name,
    wb_model_name,
    wb_project,
    save_path,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon, 
    cnn_layers, 
    cnn_filter_size,
    cnn_kernel_size, 
    cnn_dense_units,   
    metrics,
    loss,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    results,
    all_results,
    num_rounds=5
):
    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)
            
            # Initialize wandb
            wandb.init(project=wb_project_name, 
                name=f'{wb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wb_model_name,
            })

            model = build_cnn_model(X_train[f'user{idx+1}'], horizon, cnn_layers, cnn_filter_size, cnn_kernel_size, cnn_dense_units, batch_size)
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer')   

            model.save(f'{save_path}/wandb/{wb_project}_{wb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()
        
        
    for idx in range(len(df_array)):
        new_row = {
            'architecture': "cnn",
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


def run_transformer_model(
    wb_project_name,
    wb_model_name,
    wb_project,
    save_path,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon, 
    sequence_length, 
    transformer_layers, 
    num_features, 
    transformer_heads, 
    transformer_dense_units,  
    metrics,
    loss,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    results,
    all_results,
    num_rounds=5
): 
    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wb_project_name, 
                name=f'{wb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wb_model_name,
            })
            
            model = build_transformer_model(X_train[f'user{idx+1}'], horizon, batch_size, sequence_length, transformer_layers, num_features, transformer_heads, transformer_dense_units)
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer')   

            model.save(f'wandb/{wb_project}_{wb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()
        
        
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


def run_soft_dense_moe_model (
    wb_project_name,
    wb_model_name,
    wb_project,
    save_path,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon, 
    dense_smoe_units, 
    dense_smoe_num_experts, 
    dense_smoe_expert_units, 
    metrics,
    loss,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    results,
    all_results,
    num_rounds=5,
    use_new_loss=True,
):
    wandb_project_name = wb_project_name
    wandb_model_name = wb_model_name
    wandb_project = wb_project

    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wandb_project_name, 
                name=f'{wandb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wandb_model_name,
            })
            if use_new_loss:
                model = build_soft_dense_moe_model(X_train[f'user{idx+1}'], batch_size, horizon,dense_smoe_units, dense_smoe_expert_units, dense_smoe_num_experts, metrics)
            else:
                model = build_soft_dense_moe_model(X_train[f'user{idx+1}'], batch_size, horizon,dense_smoe_units, dense_smoe_expert_units, dense_smoe_num_experts, metrics, use_loss=False)

            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wandb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer')   

            model.save(f'{save_path}/wandb/{wandb_project}_{wandb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()

    dataframe = all_results     
    for idx in range(len(df_array)):
        new_row = {
            'architecture': wandb_model_name,
            'train_time': dataframe[dataframe["user"]==f"user{idx+1}"]["train_time"].mean(), 
            'avg_time_epoch' : dataframe[dataframe["user"]==f"user{idx+1}"]["avg_time_epoch"].mean(),
            'mse': dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].mean(),
            'mse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].std(),
            'rmse': dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].mean(),
            'rmse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].std(),
            'mape': dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].mean(),
            'mape_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].std(),
            'mae': dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].mean(),
            'mae_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].std(),
        }
        results.loc[len(results)] = new_row

    print(results.head(1))
    results.to_csv(f'{save_path}/wandb/{wandb_model_name}_results.csv')
    all_results.to_csv(f'{save_path}/wandb/{wandb_model_name}_all_results.csv')

def run_topk_dense_moe_model(
    wb_project_name,
    wb_model_name,
    wb_project,
    save_path,
    df_array,
    max_epochs,
    batch_size,
    X_train,
    horizon, 
    dense_topmoe_units, 
    dense_topmoe_num_experts,
    dense_topmoe_top_k,
    dense_topmoe_expert_units,
    metrics,
    loss,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    callbacks,
    results,
    all_results,
    num_rounds=5,
    use_new_loss=True,
):
    wandb_project_name = wb_project_name
    wandb_model_name = wb_model_name
    wandb_project = wb_project

    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wandb_project_name, 
                name=f'{wandb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wandb_model_name,
            })

            if use_new_loss:
                model = build_topk_dense_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, dense_topmoe_units, dense_topmoe_num_experts, dense_topmoe_top_k, dense_topmoe_expert_units, metrics)
            else:
                model = build_topk_dense_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, dense_topmoe_units, dense_topmoe_num_experts, dense_topmoe_top_k, dense_topmoe_expert_units, metrics, use_loss=False)
                        
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wandb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer') #----------------UPDATE

            model.save(f'{save_path}/wandb/{wandb_project}_{wandb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()

    dataframe = all_results 
    for idx in range(len(df_array)):
        new_row = {
            'architecture': wandb_model_name,
            'train_time': dataframe[dataframe["user"]==f"user{idx+1}"]["train_time"].mean(), 
            'avg_time_epoch' : dataframe[dataframe["user"]==f"user{idx+1}"]["avg_time_epoch"].mean(),
            'mse': dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].mean(),
            'mse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].std(),
            'rmse': dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].mean(),
            'rmse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].std(),
            'mape': dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].mean(),
            'mape_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].std(),
            'mae': dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].mean(),
            'mae_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].std(),
        }
        results.loc[len(results)] = new_row 

    print(results.head(1))
    results.to_csv(f'{save_path}/wandb/{wandb_model_name}_results.csv') 
    all_results.to_csv(f'{save_path}/wandb/{wandb_model_name}_all_results.csv') 


def run_soft_lstm_moe_model(
        wb_project_name,
        wb_model_name,
        wb_project,
        save_path,
        df_array,
        max_epochs,
        batch_size,
        X_train,
        horizon, 
        lstm_smoe_units, 
        lstm_smoe_num_experts, 
        lstm_smoe_expert_units, 
        metrics,
        loss,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        callbacks,
        results,
        all_results,
        num_rounds=5,
        use_new_loss=True,
):
    wandb_project_name = wb_project_name 
    wandb_model_name = wb_model_name 
    wandb_project = wb_project

    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wandb_project_name, 
                name=f'{wandb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wandb_model_name,
            })
            
            if use_new_loss:
                model = build_soft_biLSTM_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, lstm_smoe_units, lstm_smoe_num_experts, lstm_smoe_expert_units, metrics)
            else:
                model = build_soft_biLSTM_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, lstm_smoe_units, lstm_smoe_num_experts, lstm_smoe_expert_units, metrics, use_loss=False)
                           
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer') #----------------UPDATE

            model.save(f'{save_path}/wandb/{wandb_project}_{wandb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()

    dataframe = all_results #----------------UPDATE
    for idx in range(len(df_array)):
        new_row = {
            'architecture': wb_model_name,
            'train_time': dataframe[dataframe["user"]==f"user{idx+1}"]["train_time"].mean(), 
            'avg_time_epoch' : dataframe[dataframe["user"]==f"user{idx+1}"]["avg_time_epoch"].mean(),
            'mse': dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].mean(),
            'mse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].std(),
            'rmse': dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].mean(),
            'rmse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].std(),
            'mape': dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].mean(),
            'mape_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].std(),
            'mae': dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].mean(),
            'mae_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].std(),
        }
        results.loc[len(results)] = new_row #----------------UPDATE

    print(results.head(1)) #----------------UPDATE
    results.to_csv(f'{save_path}/wandb/{wandb_model_name}_results.csv') #----------------UPDATE
    all_results.to_csv(f'{save_path}/wandb/{wandb_model_name}_all_results.csv') #----------------UPDATE


def run_topk_lstm_moe_model(
        wb_project_name,
        wb_model_name,
        wb_project,
        save_path,
        df_array,
        max_epochs,
        batch_size,
        X_train,
        horizon, 
        lstm_topmoe_units, 
        lstm_topmoe_num_experts,
        lstm_topmoe_top_k,
        lstm_topmoe_expert_units,
        metrics,
        loss,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        callbacks,
        results,
        all_results,
        num_rounds=5,
        use_new_loss=True,
):

    wandb_project_name = wb_project_name
    wandb_model_name = wb_model_name 
    wandb_project = wb_project 

    #For each of the users
    for idx in range(len(df_array)):
        print("-----User: ", idx+1)
        for round in range(num_rounds):
            print("Round: ", round)

            # Initialize wandb
            wandb.init(project=wandb_project_name, 
                name=f'{wandb_model_name}_u{idx+1}_rd{round+1}',
                config={
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'optimizer': "Adam",
                'learning_rate': 0.001,
                'architecture': wandb_model_name,
            })

            if use_new_loss:
                model = build_topk_bilstm_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, lstm_topmoe_units, lstm_topmoe_num_experts, lstm_topmoe_top_k, lstm_topmoe_expert_units, metrics)
            else:
                model = build_topk_bilstm_moe_model(X_train[f'user{idx+1}'], batch_size, horizon, lstm_topmoe_units, lstm_topmoe_num_experts, lstm_topmoe_top_k, lstm_topmoe_expert_units, metrics, use_loss=False)
            
            histroy, user_results = compile_fit_evaluate_model(
                model=model, 
                loss=loss, 
                metrics=metrics, 
                X_train=X_train[f'user{idx+1}'],
                y_train = y_train[f'user{idx+1}'], 
                max_epochs = max_epochs, 
                batch_size=batch_size, 
                X_val=X_val[f'user{idx+1}'], 
                y_val=y_val[f'user{idx+1}'], 
                X_test=X_test[f'user{idx+1}'], 
                y_test=y_test[f'user{idx+1}'], 
                callbacks=callbacks, 
                user=f'user{idx+1}', 
                hyper=wandb_model_name,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            )
            # Add the 'architecture' column from dense_user_results to dense_results
            all_results = pd.merge(all_results, user_results, how='outer')

            model.save(f'{save_path}/wandb/{wandb_project}_{wandb_model_name}_u{idx+1}_rd{round+1}.keras')     
            print("saved model") 
            
            wandb.finish()

    dataframe = all_results
    for idx in range(len(df_array)):
        new_row = {
            'architecture': wandb_model_name,
            'train_time': dataframe[dataframe["user"]==f"user{idx+1}"]["train_time"].mean(), 
            'avg_time_epoch' : dataframe[dataframe["user"]==f"user{idx+1}"]["avg_time_epoch"].mean(),
            'mse': dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].mean(),
            'mse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mse"].std(),
            'rmse': dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].mean(),
            'rmse_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["rmse"].std(),
            'mape': dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].mean(),
            'mape_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mape"].std(),
            'mae': dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].mean(),
            'mae_std' : dataframe[dataframe["user"]==f"user{idx+1}"]["mae"].std(),
        }
        results.loc[len(results)] = new_row 

    print(results.head(1)) 
    results.to_csv(f'{save_path}/wandb/{wandb_model_name}_results.csv') 
    all_results.to_csv(f'{save_path}/wandb/{wandb_model_name}_all_results.csv') 