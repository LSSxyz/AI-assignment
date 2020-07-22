import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# 单入单出
def Get_single(path):
    res = ""
    for i in path:
        res+=i
        if i=='\\':
            res+='\\'
    # print(res)
    data = pd.read_csv(res)
    x = data['X'].to_numpy()
    y = data['Y'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # print("xtrain=", x_train)
    return x_train, x_test, y_train, y_test

# 多入单出
def Get_multiple(path):
    res = ""
    for i in path:
        res+=i
        if i=='\\':
            res+='\\'
    data = pd.read_csv(res)
    key_list = data.keys().values
    x_list = []
    y_list = []
    for key in key_list:
        # print(key.find('X'), 'ke=',key)
        if key != key_list[-1] and key.find('X')==0: 
            x_list.append(data[key].to_numpy())
        if key.find('X') == -1:
            y_list.append(data[key].to_numpy())

    # y = data[key_list[-1]].to_numpy()

    x_list = np.array(x_list).T
    y_list = np.array(y_list).T
    # print(x_list.shape) # (样本个数,输入变量个数)
    # print(y_list.shape) # (样本个数,输出变量个数)
    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.3)
    return x_train, x_test, y_train, y_test, x_list, y_list
# Get_multiple('MIMO.csv')
# 可视化训练集和测试集
def Show_single(path):
    x_train, x_test, y_train, y_test = Get_single(path)
    plt.figure(figsize=(12, 8))
    plt.title("SISO", fontsize=20)
    plt.scatter(x_train, y_train, color='red')
    plt.scatter(x_test, y_test, color='blue')
    plt.show()
# Show_single()

# 可视化训练集和测试集
def Show_multiple(path):
    x_train, x_test, y_train, y_test, x, y = Get_multiple(path)
    sz = x_train.shape[1]
    # 首先，画出每一维度的样本分布
    plt.figure(1, figsize=(100, 4))
    plt.suptitle('Sample distribution')
    for i in range(sz):
        plt.subplots_adjust(hspace=0.5, wspace=0.5, )
        plt.subplot(1, sz, i+1)
        plt.title('The {} dim'.format(i+1))
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xlabel('Sample-th')
        plt.ylabel('X-value')
        xx = np.linspace(1, x.shape[0], x.shape[0])
        plt.scatter(xx, x[:, i])
    plt.show()
    # 如果样本可以显示在三维空间，就显示
    if sz==2 and y_test.shape[1]==1:
        fig = plt.figure(2, figsize=(12, 8))
        ax=Axes3D(fig)
        x1_train, x1_test = x_train[:, 0], x_test[:, 0]
        x2_train, x2_test = x_train[:, 1], x_test[:, 1]
        x1_train, x2_train = np.meshgrid(x1_train, x2_train)
        y_train = np.add(x1_train, x2_train)
        # print(y_train.shape, x1_train)
        ax.scatter(x1_train, x2_train, y_train)
        
        x1_test, x2_test = np.meshgrid(x1_test, x2_test)
        y_test = np.add(x1_test, x2_test)
        print(y_test.shape)
        ax.scatter(x1_test, x2_test, y_test)

        ax.set_title('MISO')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        plt.show()

