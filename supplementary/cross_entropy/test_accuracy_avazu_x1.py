import lightgbm as lgb
import time
import os
import logging
logging.basicConfig(
    filename="Avazu_x1_cross_entropy.log", 
    level=logging.INFO
)

# Parameters are borrowed from the official experiment doc:
# https://lightgbm.readthedocs.io/en/latest/Experiments.html
params = {"max_bin" : 255,
"num_leaves" : 250,
"num_iterations" : 1000,
#"objective" : "binary",
"learning_rate" : 0.1,
"bagging_fraction": 0.8,         
"feature_fraction": 0.8,         
"tree_learner" : "serial",
"task":"train",
"num_thread":64,
"min_data_in_leaf":1,
"metric": "auc",
"min_sum_hessian_in_leaf":100,
}

# Note that we have convert the original raw data into a pure libsvm format.
# # For more details, pls refer to: https://github.com/guolinke/boosting_tree_benchmarks/tree/master/data
infile_train = "../data/avazu_x1/train.libsvm"
infile_valid = "../data/avazu_x1/valid.libsvm"


# Load data
train_data = lgb.Dataset(infile_train)
valid_data = lgb.Dataset(infile_valid)


# Training
res = {}
start = time.time()
logging.info("Lightgbm traning starting (AUC, avazu)...")
bst = lgb.train(
    params, train_data, 
    valid_sets=[valid_data], valid_names=["valid"],
    num_boost_round=50,
    callbacks=[lgb.record_evaluation(res), lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=10)])
end = time.time()

# Logging
logging.info(bst.best_score['valid'])
logging.info(f"this process took: {round(end - start, 2)}, seconds")