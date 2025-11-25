import lightgbm as lgb
import time
import os
import logging
logging.basicConfig(
    filename="KDD12_cross_entropy.log", 
    level=logging.INFO
)

#os.chdir('..')

# Parameters are borrowed from the official experiment doc:
# https://lightgbm.readthedocs.io/en/latest/Experiments.html
params = {   
    "metric": "auc",
    "num_leaves": 2048, 
    "max_depth": -1,
    "num_iterations": 3000,
    "learning_rate": 0.01, 
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l2": 0.1,
    "num_thread": -1,
    "max_bin": 255,
}


# Note that we have convert the original raw data into a pure libsvm format.
# # For more details, pls refer to: https://github.com/guolinke/boosting_tree_benchmarks/tree/master/data
infile_train = "..data/KDD12/train.libsvm"
infile_valid = "../data/KDD12/valid.libsvm"


# Load data
train_data = lgb.Dataset(infile_train)
valid_data = lgb.Dataset(infile_valid)


# Training
res = {}
start = time.time()
logging.info("Lightgbm traning starting (AUC, HIGGS)...")
bst = lgb.train(
    params, train_data, 
    valid_sets=[valid_data], valid_names=["valid"],
    num_boost_round=50,
    callbacks=[lgb.record_evaluation(res), lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=10)])
end = time.time()

# Logging
logging.info(bst.best_score['valid'])
logging.info(f"this process took: {round(end - start, 2)}, seconds")