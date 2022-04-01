#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2021.08.02
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : ATGPred-FL.py

import read_fasta_sequences
import numpy as np
import pandas as pd
import joblib
from feature_scripts.feature import get_proba_feature
import argparse
import time
import os
import warnings
warnings.filterwarnings('ignore')

if not os.path.exists("results"):
    os.makedirs("results")

def predict(feature):
    scale = joblib.load('./models/scaler.pkl')
    feature = scale.transform(feature)
    model = joblib.load('./models/LGBM_predict.pkl')
    y_pred_prob = model.predict_proba(feature)
    df_out = pd.DataFrame(np.zeros((y_pred_prob.shape[0], 3)), columns=["Sequence_name", "Prediction", "probability"])
    y_pred = model.predict(feature)

    for i in range(y_pred.shape[0]):
        df_out.iloc[i, 0] = str(sequence_names[i])
        if y_pred[i] == 1:
            df_out.iloc[i, 1] = "Channel Protein"
            df_out.iloc[i, 2] = "%.2f%%" % (y_pred_prob[i, 1] * 100)
        if y_pred[i] == -1:
            df_out.iloc[i, 1] = "Non-channel Protein"
            df_out.iloc[i, 2] = "%.2f%%" % (y_pred_prob[i, 0] * 100)
    os.chdir(".\Results")
    df_out.to_csv(args.o,)
    print("Job finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="Sequence-Based Prediction of Autophagy Proteins with Feature Representation Learning")
    parser.add_argument("-i", required=True, default=None, help="input fasta file")
    parser.add_argument("-o", default="Results.csv", help="output a CSV results file")
    args = parser.parse_args()

    time_start = time.time()
    print("Sequence checking......")
    sequence, sequence_names = read_fasta_sequences.read_protein_sequences(args.i)

    print("Sequence encoding......")
    encodings = get_proba_feature(sequence)

    print("Predicting......")
    predict(encodings)

    time_end = time.time()
    print('Total time cost', time_end - time_start, 'seconds')
