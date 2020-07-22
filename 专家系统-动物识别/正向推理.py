import numpy as np 
import pandas as pd 

PATH = "知识库.txt"
def Get_G(path=PATH):
    # 产生式左部和右部逐行对应
    factor = [] # 产生式左部
    result = [] # 产生式右部
    with open(path, encoding='utf-8') as file:
        f = file.readlines()
        for line in f:
            line = line.strip('\n')
            if line:
                data = line.split(' ')
                factor.append(data[:-1])
                result.append(data[-1])
    # print(
    #     factor, 
    #     '\n', 
    #     result,
    # )
    for i in range(len(factor)):factor[i].sort()
    return factor, result
class Inference():
    def __init__(self,):
        # Inference.__init__(self, input)
        self.factor, self.result = Get_G(PATH)
    # 获取终结符号
    def Get_t(self, ):
        terminal = []
        for i in range(len(self.result)):
            flag=0
            for j in range(len(self.factor)):
                if self.factor[j].count(self.result[i]):
                    flag=1
                    break
            if flag==0:terminal.append(self.result[i])
        return terminal
    # 从arr数组中抽出非终结符（只出现在产生式左部）和终结符
    def Split(self, t, arr):
        nonterminal = []
        terminal = []
        for i in range(len(arr)):
            if t.count(arr[i]):
                terminal.append(arr[i])
            else:
                nonterminal.append(arr[i])
        # print(
        #     nonterminal, 
        #     '\n', 
        #     terminal,
        # )
        return terminal, nonterminal
    # 判断输入的元素是否全部在某一条产生式的左部
    def issame(self, input, f):
        # for i in range(len(self.factor)):
        comlist = list(set(input).intersection(set(f)))
        comlist.sort()
        # print('comlist=', comlist, 'fac=', f)
        if comlist == f: return True
        
        return False

    def inference(self, input):
        # print('factor=', self.factor)
        # print(self.result)
        # print(input)
        terminal = self.Get_t()
        input_list = []
        ans_terminal = []
        input_list.append(input)
        # 特判鸵鸟 企鹅
        if input.count('会飞') and input.count('不会飞'):
            ans_terminal.append(['Wrong'])
        else:
            for i in range(len(self.factor)):
                # 如果满足条件
                if self.issame(input, self.factor[i]):
                    new = input+[self.result[i]]
                    input_list.append(new)
                    input = new
                    t, nont = self.Split(terminal, input)
                    if len(t):
                        ans_terminal.append(t[0])
                        break
            if len(ans_terminal)==0 : ans_terminal.append(['Wrong'])
        return input_list, ans_terminal
