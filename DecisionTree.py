#!/usr/bin/python3

import math
import os


class TNode:
    def __init__(self, parent, child, A, label):
        self.parent = parent
        # 子树
        self.child = child
        # 选取的特征值
        self.A = A
        # 标记分类
        self.label = label


class DecisionTree:
    __Tree = None
    __isSelected = []

    def trainModel(self, data):
        self.__Tree = self.buildTree(data, None, None)

    # 递归建立树
    def buildTree(self, data, parent, label):

        (A, H, IG, feature) = self.selectFeature(data)

        # 信息熵为0，数据只有一类
        if H == 0:
            return TNode(parent, None, None, data[0][0])

        child = []
        ds = {}

        # 父节点，所有子节点，选取的特征维度，父节点选取的特征的一个边
        node = TNode(parent, child, A, label)

        for f in feature:
            if f not in ds:
                ds[f] = []

        for item in data:
            ds[item[A]].append(item)

        for d in ds:
            c = self.buildTree(ds[d], node, d)
            node.child.append(c)

        return node

    # 选择特征值
    def selectFeature(self, data):
        HX = self.computeHX(data)
        (A, H, IG, feature) = self.computeAllHXY(data, HX)
        # 记录维度已作为分支点
        self.__isSelected.append(A)
        return A, H, IG, feature

    # 经验熵 empirical entropy
    def computeHX(self, data):

        H = 0

        allNum = len(data)
        tmpTotal = {}

        for j in range(allNum):
            key = str(data[j][0])
            if key in tmpTotal:
                tmpTotal[key] = tmpTotal[key] + 1
            else:
                tmpTotal[key] = 1

        for key in tmpTotal:
            p = float(tmpTotal[key]) / allNum
            H += -p * math.log(p, 2)

        return H

    # 经验条件熵 empirical conditional entropy
    def computeAllHXY(self, data, HX):

        H = 0
        IG = 0
        A = 0
        Feature = None
        length = len(data)

        for i in range(1, len(data[0])):

            if i in self.__isSelected:
                continue

            H = 0

            tmpFTotal = {}
            tmpF = {}
            tmpL = {}
            tmpFeature = []
            tmpLabel = []

            for j in range(length):

                f = data[j][i]

                key = str(f) + "|" + str(data[j][0])
                if key in tmpF:
                    tmpF[key] = tmpF[key] + 1
                else:
                    tmpF[key] = 1

                if f in tmpFTotal:
                    tmpFTotal[f] = tmpFTotal[f] + 1
                else:
                    tmpFTotal[f] = 1

                label = data[j][0]
                if label in tmpL:
                    tmpL[label] = tmpL[label] + 1
                else:
                    tmpL[label] = 1

                if label not in tmpLabel:
                    tmpLabel.append(label)

                if f not in tmpFeature:
                    tmpFeature.append(f)

            for feature in tmpFeature:
                Hf = 0
                for label in tmpLabel:
                    key = str(feature) + "|" + str(label)

                    if key not in tmpF:
                        tmpF[key] = 0

                    p = tmpF[key] / tmpFTotal[feature]

                    if p != 0:
                        Hf += -p * math.log(p, 2)

                H += tmpFTotal[feature] / float(length) * Hf

            if (HX - H) > IG:
                IG = HX - H
                A = i
                Feature = tmpFeature

        return A, H, IG, Feature

    # 根据决策树进行预测
    def predict(self, feature):
        label = self.traverseTree(self.__Tree, feature)
        return label

    # 遍历树
    def traverseTree(self, node, feature):

        if node.child is None:
            return node.label

        A = node.A - 1
        f = feature[A] - 1

        return self.traverseTree(node.child[f], feature)

f = open(os.getcwd() + "\\data")
data = []
for line in f.readlines():
    if line is None:
        continue
    item = line.split(" ")
    newItem = []
    for i in item[:-1]:
        newItem.append(i)
    newItem.append(item[-1][:-1])
    data.append(newItem)

dt = DecisionTree()
dt.trainModel(data)
label = dt.predict([1, 1, 1, 1])
print(label)
