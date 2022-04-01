#!/usr/bin/env python
# _*_coding:utf-8_*_

import math
import re
kw = {'order': 'ACDEFGHIKLMNPQRSTVWY'}

def get_APAAC(fastas, lambdaValue=2, w=0.05, **kw):
    with open("./feature_scripts/PAAC.txt", "r") as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])
    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    encodings = []

    for i in fastas:
        sequence = re.sub('-', '', i)
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]
        encodings.append(code)
    return encodings
