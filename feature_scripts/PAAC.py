#!/usr/bin/env python
# _*_coding:utf-8_*_

import math
import re

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def get_PAAC(fastas):
    with open("./feature_scripts/PAAC.txt", "r") as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])

    AADict = {}
    for i in range(len(AA)):  # 20
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):  # llen(records) 4
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
        for n in range(1, 3):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                     range(len(sequence) - n)]) / (
                        len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + 0.05 * sum(theta)) for aa in AA]
        code = code + [(0.05 * j) / (1 + 0.05 * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings