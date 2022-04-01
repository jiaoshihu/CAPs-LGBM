#!/usr/bin/env python
# _*_coding:utf-8_*_

import sys, platform, os, re
import argparse
import numpy as np



def get_Moran(fastas,
          props=['ANDN920101', 'ARGP820101', 'ARGP820102', 'ARGP820103', 'BEGF750101', 'BEGF750102', 'BEGF750103',
                 'BHAR880101'], nlag=2, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'

    with open("./feature_scripts/AAidx.txt", "r") as f:
        records = f.readlines()[1:]
    myDict = {}
    for i in records:
        array = i.rstrip().split('\t')
        myDict[array[0]] = array[1:]

    AAidx = []
    AAidxName = []
    for i in props:
        if i in myDict:
            AAidx.append(myDict[i])
            AAidxName.append(i)
        else:
            print('"' + i + '" properties not exist.')
            return None

    AAidx1 = np.array([float(j) for i in AAidx for j in i])
    AAidx = AAidx1.reshape((len(AAidx), 20))

    propMean = np.mean(AAidx, axis=1)
    propStd = np.std(AAidx, axis=1)

    for i in range(len(AAidx)):
        for j in range(len(AAidx[i])):
            AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []

    for i in fastas:
        sequence = re.sub('-', '', i)
        code = []
        N = len(sequence)
        for prop in range(len(props)):
            xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
            for n in range(1, nlag + 1):
                if len(sequence) > nlag:
                    # if key is '-', then the value is 0
                    fenzi = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) * (
                    AAidx[prop][index.get(sequence[j + n], 0)] - xmean) for j in range(len(sequence) - n)]) / (N - n)
                    fenmu = sum(
                        [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))]) / N
                    rn = fenzi / fenmu
                else:
                    rn = 'NA'
                code.append(rn)
        encodings.append(code)
    return encodings



