import pandas as pd
import lightgbm as lgb
import time
from tqdm import tqdm
from pltreeshap import PLTreeExplainer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from copy import copy


print("Imports worked fine :) ", flush=True)


transactions_train = pd.read_parquet('drive/MyDrive/CURRENT_DIR/data/ieee_cis_fraud_train.parquet') # columns are train_features + ['isFraud']
transactions_test = pd.read_parquet('drive/MyDrive/CURRENT_DIR/data/ieee_cis_fraud_test.parquet') # columns are train_features + ['isFraud']

train_features = [f for f in transactions_train.columns if f != 'isFraud']
fraud_train = transactions_train[train_features]
fraud_test = transactions_test[train_features]

print("Finished loading Fraud Data", flush=True)

# !!!!!!!! Train RandomForestRegressor model !!!!!!!!!!!!!!!


# !!!!!!!! Train LightGBM model !!!!!!!!!!!!!!!

LIGHTGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,

    # Allow high depth and enough leaves to reach this possible depth
    "num_leaves": 2024,
    "max_depth": 10,                 
    "min_data_in_leaf": 500,        # Does provide some regulation

    # Sampling (stability + reduces overfit)
    "feature_fraction": 0.8, 
    "bagging_fraction": 0.8, 
    "bagging_freq": 1,
    
    # Practical
    "verbosity": -1,
    "seed": 42,
    "force_col_wise": True,            # often faster/safer for wide data
}

def lightgbm_model(X_train, y_train, params, num_rounds=100):
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    return lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_rounds
    )


def different_depth_lightgbm_models(trainset, y, params, num_rounds, depths):
    models = {}
    for depth in depths:
        new_params = params.copy()
        new_params['max_depth'] = depth
        models[depth] = lightgbm_model(trainset, y, new_params, num_rounds=num_rounds)
        print(f"Trained on depth {depth}")
    return models  


print("Train 100 trees LightGBMs")    
gbm_100trees = different_depth_lightgbm_models(transactions_train[train_features], transactions_train['isFraud'], LIGHTGBM_PARAMS, num_rounds=100, depths=[6,9,12,15])
print("Train 1 trees LightGBMs")
gbm_1trees = different_depth_lightgbm_models(transactions_train[train_features], transactions_train['isFraud'], LIGHTGBM_PARAMS, num_rounds=1, depths=[15])

    
# !!!!!!!! Compute Background SHAP using PLTreeShap !!!!!!!!!!!!!!!

def pltreeshap_on_models_dict(
        models, consumer_data: pd.DataFrame, background_data: pd.DataFrame
    ):
    running_times = {}
    for key, model in models.items():
        print("Depth " + str(key))
        print("Using PLTreeShap to compute Background SHAP", flush=True)
        start_time = time.time()
        explainer = PLTreeExplainer(model)
        explainer.aggregate(background_data)  # precomputes split statistics
        print(f"aggregation of Background data took " + str(time.time() - start_time), flush=True)
        explainer.shap_values(consumer_data)
        running_time = time.time() - start_time
        running_times[key] = running_time
        print(f"On Depth {key} Took: {running_time}", flush=True)
    return running_times

def pltreeshap_iv_on_models_dict(
        models, consumer_data: pd.DataFrame, background_data: pd.DataFrame
    ):
    running_times = {}
    for key, model in models.items():
        print("Depth " + str(key))
        start_time = time.time()
        explainer = PLTreeExplainer(model)
        explainer.aggregate(background_data)  # precomputes split statistics
        aggregation_time = time.time() - start_time
        print(f"aggregation of Background data took " + str(aggregation_time), flush=True)
        iv_computation_start_time = time.time()
        explainer.shap_interaction_values(consumer_data) # Running on all rows uses too much RAM and crashes the session
        iv_computation_time = time.time() - iv_computation_start_time
        running_time = time.time() - start_time
        running_times[key] = running_time
        print(f"On Depth {key} Took: {running_time}", flush=True)
    return running_times

pltreeshap_on_models_dict(
    {6: gbm_100trees[6], 9: gbm_100trees[9], 12: gbm_100trees[12], 15: gbm_1trees[15]}, # Depth 18 crashes due to RAM
    consumer_data=fraud_test, background_data=fraud_train
)

fraud_testset_sample = fraud_test.sample(10_000, random_state=42)

pltreeshap_iv_on_models_dict(
    {6: gbm_100trees[6], 9: gbm_100trees[9], 12: gbm_100trees[12]}, # depth 15 crashes due to RAM
    consumer_data=fraud_testset_sample, background_data=fraud_train
)