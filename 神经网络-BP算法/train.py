import numpy as np 
from Network import Net
from layer import Layer
from 数据获取 import Get_single, Get_multiple
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


'''
网络结构，训练集，测试集，学习率，迭代次数
'''
def train(net, x_train, x_test, y_train, y_test, lr, epoches):
    train_losses = []
    test_losses = []
    plt.ion()
    print_interval = int(epoches*0.2)
    for i in range(epoches):
        print('i=', i)
        # 为了得到更好的效果，适当改变学习率
        if i%print_interval==0 and i!=0:
            lr=lr*0.8
            print('-----------learning rate=',lr,'------------')
        for j in range(len(x_train)):
            net.back_propagation(x_train[j], y_train[j], lr)
            if i%print_interval==0:
                # 均方差
                train_loss = np.mean(np.square(y_train - net.forward_propagation(x_train)))
                test_loss = np.mean(np.square(y_test - net.forward_propagation(x_test)))
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print('Epoch=', i, ', train_loss=', train_loss, ', test_loss=', test_loss)
                # 回想泛化实况
                plt.figure(1)
                plt.cla()
                xx = np.linspace(1, len(test_losses), len(test_losses))
                plt.title("generalization")
                plt.xlabel('epoches')
                plt.ylabel('test_loss')
                plt.plot(xx, test_losses)
                # plt.show()

                plt.figure(2)
                plt.cla()
                xx = np.linspace(1, len(train_losses), len(train_losses))
                plt.title("recall")
                plt.xlabel('epoches')
                plt.ylabel('train_loss')
                plt.plot(xx, train_losses)

                # 二维空间函数拟合
                if len(x_train[0])==1:
                    plt.figure(3)
                    plt.cla()
                    plt.scatter(x_train, y_train, s=5)
                    plt.scatter(x_train,net.forward_propagation(x_train),s=4 , color='red')
                    plt.text(1.5,0.2,'train loss=%.6lf'%train_loss,fontdict={'size':15,'color':'red','style':'italic'})

                    plt.figure(4)
                    plt.cla()
                    plt.scatter(x_test, y_test, s=5)
                    plt.scatter(x_test, net.forward_propagation(x_test), s=4, color='black')
                    plt.text(plt.axis()[0]+0.1,plt.axis()[2]+0.1,'test loss=%.6lf'%test_loss,fontdict={'size':15,'color':'blue','style':'italic'})

                    plt.pause(0.01)
                # 三维空间函数拟合
                elif len(x_train[0])==2:
                    x1_train = x_train[:, 0]
                    x2_train = x_train[:, 1]
                    x1_train, x2_train = np.meshgrid(x1_train, x2_train)
                    trainY = np.add(x1_train, x2_train)
                    x1_test = x_test[:, 0]
                    x2_test = x_test[:, 1]
                    x1_test, x2_test = np.meshgrid(x1_test, x2_test)
                    testY = np.add(x1_test, x2_test)
                    a = []
                    b = []
                    
                    for k in x1_train:
                        for p in x2_train:
                            m = np.vstack((k, p)).T
                            a.append(m)
                    a = np.array(a)
                    a.resize((x1_train.shape[0]*x2_train.shape[0], 2))

                    for k in x1_test:
                        for p in x2_test:
                            m = np.vstack((k, p)).T
                            b.append(m)
                    b = np.array(b)
                    b.resize((x1_test.shape[0]*x2_test.shape[0], 2))
                    trainy =  net.forward_propagation(a).reshape((x1_train.shape[0], x2_train.shape[0]))
                    testy = net.forward_propagation(b).reshape((x1_test.shape[0], x2_test.shape[0]))
                    # print(trainy.shape, testy.shape) #(280, 280), (120, 120)

                    fig1 = plt.figure(3, figsize=(6, 4))
                    plt.cla()
                    ax1=Axes3D(fig1)
                    ax1.scatter(x1_train, x2_train, trainY, ) # 预期输出
                    ax1.scatter(x1_train, x2_train, trainy, ) # 实际输出
                    ax1.text(plt.axis()[1]-0.1, plt.axis()[2]+0.1, 0, 'train loss=%.6lf'%train_loss, fontdict={'size':15,'color':'blue','style':'italic'})
                    ax1.set_xlabel('x1')
                    ax1.set_ylabel('x2')
                    ax1.set_zlabel('y')

                    fig2 = plt.figure(4, figsize=(6, 4))
                    plt.cla()
                    ax2=Axes3D(fig2)
                    ax2.scatter(x1_test, x2_test, testY, )
                    ax2.scatter(x1_test, x2_test, testy, )
                    ax2.text(plt.axis()[1]-0.1, plt.axis()[2]+0.1, 0, 'test loss=%.6lf'%test_loss, fontdict={'size':15,'color':'blue','style':'italic'})
                    ax2.set_xlabel('x1')
                    ax2.set_ylabel('x2')
                    ax2.set_zlabel('y')
                    plt.pause(0.01)

    plt.show()
    plt.ioff()
    return train_losses, test_losses

def draw_losses(train, test):
    x=np.linspace(1, len(train), len(train))
    train = np.array(train)
    plt.figure(5)
    plt.title("train_losses", fontsize=20)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.plot(x, train)
    # plt.show()

    xx=np.linspace(1, len(test), len(test))
    test = np.array(test)
    plt.figure(6)
    plt.title("test_losses", fontsize=20)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.plot(xx, test)
    plt.show()


def GetNet(mes):
    layer = []
    act = []
    for i in mes.split():
        if i[0]>='a' and i[0]<='z' or i[0]>'A' and i[0]<='Z':
            act.append(i)
        else:
            layer.append(int(i))
    net=Net()
    for i in range(len(act)):
        net.addLayer(Layer(layer[i], layer[i+1], act[i]))
    return net

def train_and_test(mes1, mes2, selection, path):
    ''' 
        mes1: 每层结点个数
        mes2: 学习率和迭代次数
        selection: 单入单出、多入单出、多入多出
        path: 文件路径
    '''
    # 网络结构
    net = GetNet(mes1)
    # 数据集
    if selection=='单入单出':
        x_train, x_test, y_train, y_test = Get_single(path)
        x_train = np.expand_dims(x_train, 1).repeat(1, axis=1)
        x_test = np.expand_dims(x_test, 1).repeat(1, axis=1)
    elif selection=='多入单出':
        x_train, x_test, y_train, y_test, x, y = Get_multiple(path)
    # elif selection=='多入多出':
    #     x_train, x_test, y_train, y_test, x, y = Get_multiple(path)

    params = mes2.split()
    learning_rate = float(params[0])
    epoch = int(params[1])
    print(
        'lr=', learning_rate, 
        'epoch=', epoch, 
    )
    train_losses, test_losses = train(net, x_train, x_test, y_train, y_test, lr=learning_rate, epoches=epoch)
    # draw_losses(train_losses, test_losses)

# mes1="2 sigmoid 10 sigmoid 2"
# mes2="0.1 5"
# selection="多入多出"
# path="MIMO.csv"
# train_and_test(mes1, mes2,selection, path)




