#!/usr/bin/env python
# _*_coding:utf-8_*_

import sys, platform, os, re
import numpy as np
import argparse




def get_SOCNumber(fastas, nlag=3, **kw):

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open("./feature_scripts/Schneider-Wrede.txt", "r") as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array(
        [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

    with open("./feature_scripts/Grantham.txt", "r") as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array(
        [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
        (20, 20))

    encodings = []


    for i in fastas:
        sequence = re.sub('-', '', i)
        code = []
        for n in range(1, nlag + 1):
            code.append(sum(
                [AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (
                            len(sequence) - n))

        for n in range(1, nlag + 1):
            code.append(sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                             range(len(sequence) - n)]) / (len(sequence) - n))
        encodings.append(code)
    return encodings



