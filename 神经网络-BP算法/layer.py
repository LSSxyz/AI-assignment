# 每一层的处理，目的是模拟添加网络层
import numpy as np 

# 根据每层的输入节点数n、输出节点数m、激活函数类型建立网络层，其他值默认随机生成
class Layer:
    def __init__(self, n, m, activation):
        self.activation = activation
        # 初始化权重，Xavier initialization适用于sigmoid和tanh, He initialization适用于relu
        self.w = np.random.randn(n, m)*np.sqrt(2/n) if activation=='relu' else np.random.randn(n, m)*np.sqrt(1/n)
        # self.w = np.random.randn(n, m) * np.sqrt(1 / m) 
        # 初始化偏置
        # self.b = np.zeros((m, 1))
        self.b = np.random.rand(m) * 0.1
        self.o = None # 网络层的输出
        self.g = None # 保存激活函数的导数
        # 反向传播时每层的误差参数
        self.alpha = None
        self.beta = None
    # 经过激活函数之前的计算结果
    def Get_h(self, x):
        h = np.dot(x, self.w) + self.b
        # print('x size=', x.shape, 'weight size=', self.w.shape, 'b size=', self.b.shape)
        return h 

    # 经过激活函数之后的网络层计算结果
    def Get_o(self, x):
        h = self.Get_h(x)
        # print('before r =', h.shape)
        if self.activation=='sigmoid':
            self.o = 1/(1+np.exp(-h))
        elif self.activation=='relu':
            self.o = np.maximum(h, 0)
        elif self.activation=='tanh':
            self.o = np.tanh(h)
        # 如果不是以上三种激活函数，就默认没有激活函数
        else:
            self.o = h
        # print('layer out size =', self.o.shape)
        return self.o
    # 计算激励函数的导数
    def Get_g(self, x):
        if self.activation=='sigmoid':
            self.g = x*(1-x)
        elif self.activation=='tanh':
            self.g = 1-x**2
        elif self.activation=='relu':
            self.g = x.copy()
            self.g[x>0] = 1
            self.g[x<0] = 0
        # 如果不是以上三种激励函数，默认没有激励函数，导数是1
        else:
            self.g = np.ones_like(x)
        return self.g

    

# import numpy as np 
# from 数据获取 import Get_sin, Get_sum
# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D
# class Layer:
#     def __init__(self, n_input, n_output, activation):
#         self.w = np.random.randn(n_input, n_output) * np.sqrt(1 / n_output) 
#         self.b = np.random.rand(n_output) * 0.1
#         self.activation = activation # 激活函数类型，如’sigmoid’         
#         self.o = None # 激活函数的输出值 o         
#         self.alpha = None  # 用于计算当前层的 beta 变量的中间变量 
#         self.beta = None  # 记录当前层的 beta 变量，用于计算梯度 
    
#     def Get_out(self, x):
#         # 前向计算函数
#         r = np.dot(x, self.w) + self.b # X@W + b
#         print('x size =', x.shape, 'weight size=', self.w.shape, 'b size=', self.b.shape)
#         print('before r =', r.shape)
#         # 通过激活函数，得到全连接层的输出 o (o)      
#         self.o = self.After_act(r) 
#         print('layer output size=', self.o.shape)
#         return self.o
    
#     def After_act(self, r): # 计算激活函数的输出
#         if self.activation is None:
#             return r # 无激活函数，直接返回
#         elif self.activation == 'relu':
#             return np.maximum(r, 0)
#         elif self.activation == 'tanh':
#             return np.tanh(r)
#         elif self.activation == 'sigmoid':
#             return 1 / (1 + np.exp(-r))
        
#         return r
    
#     def Get_grad(self, r):
#         # 计算激活函数的导数
#         # 无激活函数， 导数为 1
#         if self.activation is None:
#             return np.ones_like(r)
#         # ReLU 函数的导数
#         elif self.activation == 'relu':             
#             grad = np.array(r, copy=True)             
#             grad[r > 0] = 1.             
#             grad[r <= 0] = 0.             
#             return grad
#         # tanh 函数的导数实现         
#         elif self.activation == 'tanh':             
#             return 1 - r ** 2 
#         # Sigmoid 函数的导数实现         
#         elif self.activation == 'sigmoid': 
#             return r * (1 - r)
#         return r
# class Net:
#     def __init__(self):
#         self.layers = [] # 网络层对象列表
    
#     def add_layer(self, x):
#         self.layers.append(x)
    
#     def forward_propagation(self, x):
#         # 前向传播（求导）
#         for layer in self.layers:
#             x = layer.Get_out(x)
#         return x
    
#     def back_propagation(self, x, y, lr):
#         # 前向传播的实际输出是o
#         o = self.forward_propagation(x)
#         # 找出最后一层
#         last_layer = self.layers[-1]
#         last_layer.alpha =  y-o
#         last_layer.beta = last_layer.alpha*last_layer.Get_grad(o)
#         # 剩余层
#         for i in reversed(range(len(self.layers))):
#             if self.layers[i] == self.layers[-1]:continue 
#             # 倒数第二层
#             if self.layers[i+1] == self.layers[-1]:
#                 layer = self.layers[i]
#                 layer.alpha = np.dot(last_layer.w, last_layer.beta)
#                 layer.beta = layer.alpha*layer.Get_grad(layer.o)
#             # 剩余前面层
#             else:
#                 now_layer = self.layers[i]
#                 nxt_layer = self.layers[i+1]
#                 now_layer.alpha = np.dot(nxt_layer.w, nxt_layer.beta)
#                 now_layer.beta = now_layer.alpha*now_layer.Get_grad(now_layer.o)
#         for i in range(len(self.layers)):
#             now_layer = self.layers[i]
#             pre_out = np.atleast_2d(x)
#             if i:
#                 pre_out = np.atleast_2d(self.layers[i-1].o)
#             # if pre_out.ndim<2:
#             #     pre_out = pre_out.reshape(pre_out.shape[0], 1)


#             # print('pre_out size=', pre_out.shape)
#             # pre_out size= (1, 2)
#             # pre_out size= (10,)
#             # if i != 0:
#             #     pre_out = self.layers[i-1].o.T
#             # else:
#             #     # 方便后面计算
#             #     pre_out = np.atleast_2d(x).T
#             # pre_out = np.atleast_2d(x if i == 0 else self.layers[i - 1].o)
#             print('pre_out size=', pre_out.shape)
#             # pre_out size= (1, 2)
#             # pre_out size= (1, 10)            
#             now_layer.w += now_layer.beta*pre_out.T*lr

#         # # 反向传播算法实现
#         # # 向前计算，得到最终输出值
#         # output = self.forward_propagation(x)
#         # for i in reversed(range(len(self.layers))): # 反向循环
#         #     layer = self.layers[i]
#         #     if layer == self.layers[-1]: # 如果是输出层
#         #         layer.alpha = y - output
#         #         # 计算最后一层的 beta，参考输出层的梯度公式
#         #         layer.beta = layer.alpha * layer.Get_grad(output)
#         #     else: # 如果是隐藏层
#         #         next_layer = self.layers[i + 1]
#         #         layer.alpha = np.dot(next_layer.w, next_layer.beta)
#         #         layer.beta = layer.alpha*layer.Get_grad(layer.o)
        
#         # # 循环更新权值
#         # for i in range(len(self.layers)):
#         #     layer = self.layers[i]
#         #     # o_i 为上一网络层的输出
#         #     o_i = np.atleast_2d(x if i == 0 else self.layers[i - 1].o)
#         #     # 梯度下降算法，beta 是公式中的负数，故这里用加号 
#         #     layer.w += layer.beta * o_i.T * lr 
    
#     def train(self, X_train, X_test, y_train, y_test, lr, max_epochs):
#         # 网络训练函数
#         # one-hot 编码
#         # y_onehot = np.zeros((y_train.shape[0], 2)) 
#         # y_onehot[np.arange(y_train.shape[0]), y_train] = 1
#         mses = [] 
#         losshis=[]
#         plt.ion()
#         for i in range(max_epochs):  
#             if i%50==0 and i!=0:
#                 lr = lr * 0.8        
#             for j in range(len(X_train)):  # 一次训练一个样本  
#                 print('xtrain size=', X_train[j].shape)              
#                 self.back_propagation(X_train[j], y_train[j], lr)
#                 print('i=', i)             
#                 if i%50==0:
#                     # 打印出 MSE Loss                 
#                     mse = np.mean(np.square(y_train - self.forward_propagation(X_train)))    
#                     loss = np.mean(np.square(y_test - self.forward_propagation(X_test)))    
#                     mses.append(mse)   
#                     losshis.append(loss)              
#                     print('Epoch: #%s, MSE: %f, LOSS: %f, Accuracy: %.2f%%' % 
#                           (i, float(mse), float(loss), self.accuracy(self.predict(X_test), y_test.flatten()) * 100)) 
#                     if len(X_train[0])==1:
#                         plt.figure(1)
#                         plt.cla()
#                         plt.scatter(X_train, y_train, s=5)
#                         plt.scatter(X_train,self.forward_propagation(X_train),s=4 , color='red')
#                         plt.text(1.5,0.2,'train loss=%.6lf'%mse,fontdict={'size':15,'color':'red','style':'italic'})

#                         plt.figure(2)
#                         plt.cla()
#                         plt.scatter(X_test, y_test, s=5)
#                         plt.scatter(X_test, self.forward_propagation(X_test), s=4, color='black')
#                         plt.text(1.5,0.2,'test loss=%.6lf'%loss,fontdict={'size':15,'color':'blue','style':'italic'})
#                         print(loss)

#                         # plt.plot(X_test,self.forward_propagation(X_test),'m-', lw=4)
#                         # plt.text(0,-0.5,'loss=%.6lf'%mse,fontdict={'size':15,'color':'red','style':'italic'})
#                         # print(loss)
#                         plt.pause(0.01)
#                     else:
#                         x1_train = X_train[:, 0]
#                         x2_train = X_train[:, 1]
#                         x1_train, x2_train = np.meshgrid(x1_train, x2_train)
#                         trainY = np.add(x1_train, x2_train)
#                         x1_test = X_test[:, 0]
#                         x2_test = X_test[:, 1]
#                         x1_test, x2_test = np.meshgrid(x1_test, x2_test)
#                         testY = np.add(x1_test, x2_test)
#                         a = []
#                         b = []
                       
#                         for k in x1_train:
#                             for p in x2_train:
#                                 m = np.vstack((k, p)).T
#                                 a.append(m)
#                         a = np.array(a)
#                         a.resize((x1_train.shape[0]*x2_train.shape[0], 2))

#                         for k in x1_test:
#                             for p in x2_test:
#                                 m = np.vstack((k, p)).T
#                                 b.append(m)
#                         b = np.array(b)
#                         b.resize((x1_test.shape[0]*x2_test.shape[0], 2))
#                         trainy = self.forward_propagation(a).reshape((x1_train.shape[0], x2_train.shape[0]))
#                         testy = self.forward_propagation(b).reshape((x1_test.shape[0], x2_test.shape[0]))
#                         # print(trainy.shape, testy.shape) #(280, 280), (120, 120)

#                         fig1 = plt.figure(1, figsize=(6, 4))
#                         plt.cla()
#                         ax1=Axes3D(fig1)
#                         ax1.scatter(x1_train, x2_train, trainY, ) # 预期输出
#                         ax1.scatter(x1_train, x2_train, trainy, ) # 实际输出
#                         ax1.text(2, -2, -4, 'train loss=%.6lf'%mse, fontdict={'size':15,'color':'blue','style':'italic'})
#                         ax1.set_xlabel('x1')
#                         ax1.set_ylabel('x2')
#                         ax1.set_zlabel('y')

#                         fig2 = plt.figure(2, figsize=(6, 4))
#                         plt.cla()
#                         ax2=Axes3D(fig2)
#                         ax2.scatter(x1_test, x2_test, testY, )
#                         ax2.scatter(x1_test, x2_test, testy, )
#                         ax2.text(2, -2, -4, 'test loss=%.6lf'%mse, fontdict={'size':15,'color':'blue','style':'italic'})
#                         ax2.set_xlabel('x1')
#                         ax2.set_ylabel('x2')
#                         ax2.set_zlabel('y')
#                         plt.pause(0.01)

#         plt.ioff()
#         plt.show()
#         return mses
    
#     def accuracy(self, y_predict, y_test): # 计算准确度
#         return np.sum(y_predict == y_test) / len(y_test)
    
#     def predict(self, X_predict):
#         y_predict = self.forward_propagation(X_predict) # 此时的 y_predict 形状是 [600 * 2]，第二个维度表示两个输出的概率
#         y_predict = np.argmax(y_predict, axis=1)
#         return y_predict  

# # sum数据集
# x1_train, x1_test, x2_train, x2_test, y_train, y_test = Get_sum()
# X_train = np.vstack((x1_train, x2_train)).T
# X_test = np.vstack((x1_test, x2_test)).T
# # print(X_train[10][0], X_train[10][1], y_train[10])
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# # sin数据集
# # X_train, X_test, y_train, y_test = Get_sin()
# # X_train = np.expand_dims(X_train, 1).repeat(1, axis=1)
# # X_test = np.expand_dims(X_test, 1).repeat(1, axis=1)
# # print(X_train.shape, X_test.shape)

# nn = Net() # 实例化网络类 
# nn.add_layer(Layer(2,  10, 'sigmoid'))  # 隐藏层 1, 2=>25 
# # nn.add_layer(Layer(50, 50, 'sigmoid')) # 隐藏层 2, 25=>50 
# # nn.add_layer(Layer(50, 25, 'sigmoid')) # 隐藏层 3, 50=>25 
# nn.add_layer(Layer(10, 1, 'sigmoid'))  # 输出层, 25=>1
# nn.train(X_train, X_test, y_train, y_test, lr=0.1, max_epochs=1000)


 

