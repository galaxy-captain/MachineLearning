#!/usr/bin/python3

class NaiveBayes:
    # 先验概率
    __pre = {}
    # 后验概率
    __probabilities = {}
    # 类别总数
    __events = {}

    # 预测
    def predict(self, features):
        max = 0
        label = 0
        for y in self.__events:
            prob = self.__pre[y]
            for i in features:
                key = str(i) + "," + str(y)
                prob *= self.__probabilities[key]
            if prob > max:
                max = prob
                label = y
        print("分类后标签为：", end="")
        print(label, end="")
        print(",概率为：", end="")
        print(max)

    # 训练模型
    def trainModel(self, data):
        self.countEvent(data)
        self.computeProb(data)
        self.showModel()

    # 先验概率
    def countEvent(self, data):
        # 遍历每行数据，计算各先验事件总数
        for i in range(len(data)):
            key = str(data[i][0])
            had = self.__events.__contains__(key)
            if had:
                self.__events[key] = self.__events[key] + 1
            else:
                self.__events[key] = 1
        #计算各先验概率
        for key in self.__events.keys():
            self.__pre[key] = self.__events[key] / len(data)

    # 后验概率
    def computeProb(self, data):
        self.findAllPro(data)
        self.totalProb(data)

    # 计算所有后验概率对应概率的数量
    # __probabilities(P(X|Y),Count)
    def findAllPro(self, data):
        for y in self.__events.keys():
            for i in range(len(data)):
                item = data[i]
                if str(item[0]) == str(y):
                    for j in range(1, len(item)):
                        key = str(item[j]) + "," + str(y)
                        if self.__probabilities.__contains__(key):
                            self.__probabilities[key] = self.__probabilities[key] + 1
                        else:
                            self.__probabilities[key] = 1

    # 根据获取的各概率的数量，计算个后验概率的概率值
    # __probabilities(P(X|Y),probability)
    def totalProb(self, data):
        for key in self.__probabilities.keys():
            y = key.split(",")[1]
            self.__probabilities[key] = self.__probabilities[key] / self.__events[y]

    def showModel(self):
        print("各类别总数：", end="")
        print(self.__events)
        print("先验概率：", end="")
        print(self.__pre)
        print("后验概率：", end="")
        print(self.__probabilities)


data = [[-1, 1, 's'], [-1, 1, 'm'], [1, 1, 'm'], [1, 2, 's'], [-1, 2, 's'], [-1, 2, 'm'], [-1, 3, 'l'], [1, 3, 'm'],
        [1, 3, 'l'], [-1, 3, 'm']]

n = NaiveBayes()
n.trainModel(data)
n.predict([2, 's'])

