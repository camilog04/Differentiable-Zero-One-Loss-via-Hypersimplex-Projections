Here we include supplementary material on tabular dataset 
Steps to replicate
1) download data for each dataset and run scripts to construct train and test splits, download each of the files and run the corresponding scripts to generate .libsvm flies
    - Avazu_x1: https://huggingface.co/datasets/reczoo/Avazu_x1
    - Criteo_x1: https://huggingface.co/datasets/reczoo/Criteo_x1
    - Flight: run dataexpo.txt, create data with .py file in directory. Instructions taken from: https://github.com/guolinke/boosting_tree_benchmarks/tree/master/data)
    - Higgs: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#HIGGS
    - KKbox: https://huggingface.co/datasets/reczoo/KKBox_x1
    - KDD10: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20(bridge%20to%20algebra)
    - KKD12: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012
    - MovieLensLatest_x1: https://huggingface.co/datasets/reczoo/MovielensLatest_x1  
2) run each .py for either cross entropy loss or hypersimplex