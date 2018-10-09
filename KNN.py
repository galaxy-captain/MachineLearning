#!/usr/bin/python3

import math


class DistanceKV:

    def __init__(self, type, distance):
        self.type = type
        self.distance = distance


class KNN:
    __distanceList = []
    __k = 6
    __result = {}

    __KDTree = None
    __KDDistance = 0
    __NNode = None

    # 输出列表
    def showList(self, list):
        for item in list:
            print(item)

    # 基于遍历数据集获取欧氏距离的序列,预测特征向量的分类
    def predict(self, data, features):
        self.buildSortList(data, features)
        self.countResult()
        self.findMax()
        self.showModel()

    # 构建有序列表
    def buildSortList(self, data, features):
        for index in range(len(data)):
            length = self.euclideanDistance(data[index][1:], features)
            kv = DistanceKV(item[0], length)
            self.insertSort(kv)

    # 计算欧氏距离
    def euclideanDistance(self, item, features):
        length = 0
        for j in range(len(item)):
            length += ((item[j] - features[j]) ** 2)
        return length ** 0.5

    # 根据序列求各分类的数量
    def countResult(self):

        length = self.__k
        if len(self.__distanceList) < self.__k:
            length = len(self.__distanceList)

        for i in range(length):
            item = self.__distanceList[i]
            if self.__result.__contains__(item.type):
                self.__result[item.type] = self.__result[item.type] + 1
            else:
                self.__result[item.type] = 1

    # 输出预测结果
    def findMax(self):
        maxKey = -1
        maxValue = -1
        for key in self.__result.keys():
            if self.__result[key] > maxValue:
                maxKey = key
                maxValue = self.__result[key]

        length = self.__k
        if len(self.__distanceList) < self.__k:
            length = len(self.__distanceList)

        print("k=" + str(self.__k) + ",分类:" + str(maxKey) + ",概率:" + str(maxValue / length))

    # 插入排序
    def insertSort(self, kv):
        if self.__distanceList:
            isInsert = False
            for i in range(len(self.__distanceList)):
                item = self.__distanceList[i]
                if item.distance > kv.distance:
                    self.__distanceList.insert(i, kv)
                    isInsert = True
                    break
            if isInsert == False:
                self.__distanceList.append(kv)
        else:
            self.__distanceList.append(kv)

    # 展示模型
    def showModel(self):
        print("欧氏距离升序排列")
        for i in range(len(self.__distanceList)):
            print(str(self.__distanceList[i].type) + "," + str(self.__distanceList[i].distance))

    class TNode:
        def __init__(self, data, d, fchild):
            self.data = data
            self.d = d
            self.fchild = fchild
            self.lchild = None
            self.rchild = None

    # 训练kd树用于搜寻
    def trainModel(self, data):
        k = len(data[0]) - 1
        self.__KDTree = self.buildKDTree(k, data, 0, None)

    # 递归建立二叉树
    def buildKDTree(self, k, data, deep, fchild):

        if len(data) == 0:
            return

        index = ((deep % k) + 1)
        data = self.sortData(data, index)
        # 取中位数
        mid = int(len(data) / 2)
        # 维度减1,对应到数组下标
        tnode = self.TNode(data[mid], index - 1, fchild)

        tnode.lchild = self.buildKDTree(k, data[:mid], deep + 1, tnode)
        tnode.rchild = self.buildKDTree(k, data[mid + 1:], deep + 1, tnode)

        return tnode

    # 插入排序,以第index列数据排序
    def sortData(self, data, index):
        newData = [data[0]]
        for i in range(1, len(data)):
            isInsert = False
            for j in range(len(newData)):
                if data[i][index] < newData[j][index]:
                    isInsert = True
                    newData.insert(j, data[i])
                    break
            if isInsert == False:
                newData.append(data[i])
        return newData

    # 搜索kd树
    def searchKDTree(self, features):

        tmp = self.__KDTree
        node = tmp
        while tmp:
            node = tmp
            if (tmp.data[tmp.d + 1] > features[tmp.d]):
                tmp = tmp.lchild
            else:
                tmp = tmp.rchild
        # 特征向量和最近点的距离
        self.__KDDistance = self.euclideanDistance(features, node.data[1:])
        self.__distanceList.append(DistanceKV(str(node.data[0]), self.__KDDistance))
        # 遍历kd树
        self.travelKDTree(node, features)
        # 对结果进行排序
        self.sortDataToList()

        self.countResult()
        self.findMax()

    # 遍历kd树执行搜索规则
    def travelKDTree(self, node, features):
        if node.fchild is None:
            return
        fchild = node.fchild
        if abs(fchild.data[fchild.d + 1] - features[fchild.d]) <= self.__KDDistance:

            #  计算父节点是距离在distance之内
            distance = self.euclideanDistance(features, fchild.data[1:])
            if distance <= self.__KDDistance:
                self.__distanceList.append(DistanceKV(str(fchild.data[0]), distance))

            # 递归在子节点中查找
            if fchild.lchild is node:
                self.postTravel(fchild.rchild, features)
            if fchild.rchild is node:
                self.postTravel(fchild.lchild, features)

        # 遍历上级节点
        self.travelKDTree(node.fchild, features)

    # 后序遍历二叉树,并找到距离小于distance的点
    def postTravel(self, node, features):
        if node is None:
            return
        self.postTravel(node.lchild, features)
        self.postTravel(node.rchild, features)

        distance = self.euclideanDistance(features, node.data[1:])
        if distance <= self.__KDDistance:
            self.__distanceList.append(DistanceKV(str(node.data[0]), distance))

    # 对结果进行插入排序
    def sortDataToList(self):
        for i in range(len(self.__distanceList)):
            for j in range(i):
                if self.__distanceList[i].distance < self.__distanceList[j].distance:
                    tmp = self.__distanceList.pop(i)
                    self.__distanceList.insert(j, tmp)
                    break


f = open("C:\\Users\\duanxl1123\\Desktop\\作业\\机器学习\\flower_format.txt")
data = []
for line in f.readlines():
    item = line.split(" ")
    newItem = [item[0]]
    for i in item[1:-1]:
        newItem.append(float(i))
    data.append(newItem)

knn = KNN()
# knn.predict(data, [6.1, 2.9, 4.7, 1.4])
# knn.trainModel([[1, 2, 3], [1, 5, 4], [2, 9, 6], [2, 4, 7], [2, 8, 1], [1, 7, 2]])
# knn.searchKDTree([3, 5])
knn.trainModel(data)
# knn.searchKDTree([5.4, 3.4, 1.5, 0.4])
# knn.searchKDTree([6.1, 2.9, 4.7, 1.4])
knn.searchKDTree([7.9, 3.8, 6.4, 2])
