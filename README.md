# CAPs-LGBM

# 1 Description
Channel proteins are hydrophilic channels across the plasma membrane that allow ions of the appropriate size to pass along the concentration gradient and play an essential role in the material exchange of cells with their surrounding environment. Here, we developed the first machine learning-based software that enables the classification of proteins into channel proteins or non-channel proteins. CAPs-LGBM, which identifies channel proteins using feature representation learning scheme and lightGBM algorithm. It has the potential to facilitate future computational work in this field. Webserver and datasets available at: http://lab.malab.cn/~acy/CAPs-LGBM

# 2 Requirements
Before running, please make sure the following packages are installed in Python environment:

joblib==1.1.0  numpy==1.21.5  pandas==1.3.5  scikit-learn==1.0.2  xgboost==1.4.2  scipy==1.7.3  lightgbm==3.3.2

For convenience, we strongly recommended users to install the Anaconda Python 3.8.5 (or above) in your local computer.

# 3 Running
Changing working dir to CAPs-LGBM-main, and then running the following command:

python CAPs-LGBM.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
