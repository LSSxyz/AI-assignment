import numpy as np 
from layer import Layer

class Net:
    def __init__(self):
        # 网络的每一层
        self.layers = []
    # 添加一层网络
    def addLayer(self, x):
        self.layers.append(x)
    
    # 前向传播
    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.Get_o(x)
        return x
    
    # 反向传播
    ''' 
        x是输入，y是预期输出，lr是现在的学习率
    '''
    def back_propagation(self, x, y, lr):
        # 前向传播的实际输出是o
        o = self.forward_propagation(x)
        # 找出最后一层
        last_layer = self.layers[-1]
        last_layer.alpha =  y-o
        last_layer.beta = last_layer.alpha*last_layer.Get_g(o)
        # 剩余层
        for i in reversed(range(len(self.layers))):
            if self.layers[i] == self.layers[-1]:continue 
            # 倒数第二层
            if self.layers[i+1] == self.layers[-1]:
                layer = self.layers[i]
                layer.alpha = np.dot(last_layer.w, last_layer.beta)
                layer.beta = layer.alpha*layer.Get_g(layer.o)
            # 剩余前面层
            else:
                now_layer = self.layers[i]
                nxt_layer = self.layers[i+1]
                now_layer.alpha = np.dot(nxt_layer.w, nxt_layer.beta)
                now_layer.beta = now_layer.alpha*now_layer.Get_g(now_layer.o)
        for i in range(len(self.layers)):
            now_layer = self.layers[i]
            pre_out = np.atleast_2d(x)
            if i:
                pre_out = np.atleast_2d(self.layers[i-1].o)

            now_layer.w += now_layer.beta*pre_out.T*lr


    