#!/usr/bin/python


class Perceptron:
    __weights = []
    __offset = 0
    __rate = 0
    __maxItr = 0
    __hasItrCount = 0

    def predict(self, features):
        fx = 0
        for i in range(len(features)):
            fx += self.__weights[i] * features[i]
        fx += self.__offset
        return 1 if fx > 0 else -1

    # data第一列为label
    def trainModel(self, data, maxItr=100):
        # 初始化线性方程
        self.__maxItr = maxItr
        self.__hasItrCount = 0
        self.initFormula(len(data[0]) - 1)
        # 迭代进行训练
        index = self.checkLoss(data)
        while self.__hasItrCount < maxItr and index >= 0:
            self.updateFormula(data[index])
            index = self.checkLoss(data)
            self.__hasItrCount += 1
            print(self.__weights,self.__offset)
        # 输出最后的模型
        self.showModel()
        print(self.__hasItrCount)

    # 初始化线性方程
    def initFormula(self, weightLength):
        self.__rate = 1
        self.__offset = 0
        for i in range(weightLength):
            self.__weights.append(0)

    # 检测误差
    def checkLoss(self, data):
        for i in range(len(data)):
            item = data[i]
            # 计算每个特征值y(wx+b)值
            label = self.__offset
            for j in range(1, len(item)):
                label += self.__weights[j - 1] * item[j]
            # label小于等于0时为误差值
            if label * item[0] <= 0:
                return i
        return -1

    # 利用特征向量更新方程的参数
    def updateFormula(self, feature):
        # offset <= offset + rate * y
        self.__offset += self.__rate * feature[0]
        # weight <= weight + rate * y * feature
        for i in range(len(self.__weights)):
            self.__weights[i] += self.__rate * feature[0] * feature[i + 1]

    def showModel(self):
        print("训练后模型：f(x) = ", end="")
        for i in range(len(self.__weights)):
            if self.__weights[i] >= 0:
                print("+%s*x%s" % (str(self.__weights[i]), str(i + 1)), end='')
            else:
                print("%s*x%s" % (str(self.__weights[i]), str(i + 1)), end='')
        if self.__offset >= 0:
            print("+%s" % self.__offset)
        else:
            print("%s" % self.__offset)


data = [[1, 3, 3], [1, 4, 3], [-1, 1, 2]]

p = Perceptron()
p.trainModel(data)
label = p.predict([3, 3])
print("分类：" + str(label))
