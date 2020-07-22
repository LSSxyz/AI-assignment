import numpy as np 
import pandas as pd 
# import collections
# import operator
# import functools
from copy import deepcopy
PATH = "知识库.txt"
def Get_G(path):
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

# 获得能够识别的动物集合
def Get_animal(factor, result):
    animal = []
    for i in range(len(result)):
        flag = True
        for j in factor:
            if j.count(result[i]):
                flag = False
        if flag:animal.append(result[i])
    animal = list(set(animal))
    return animal

# 得到没有作为结论过的条件的集合
def Get_r(factor, result):
    r = []
    for item in factor:
        for i in item:
            if result.count(i) == 0:r.append(i)
    r = list(set(r))
    return r

# 得到条件都是没有作为结论过的规则的集合
def Get_rule(factor, result, r):
    rule = {}
    for i in range(len(factor)):
        flag = True
        for j in factor[i]:
            if r.count(j) == 0:
                flag = False
                break
        if flag:
            if result[i] not in rule.keys():
                rule[result[i]] = []
            rule[result[i]].append(factor[i])
    return rule

# 得到结果为res的规则的编号的集合
def Get_index(result, res):
    index = []
    for i in result:
        if i == res:index.append(i)
    return index

# 对于一个字符串，求结论是该字符串的规则的编号的集合
def getlist(result, res):
    ans = []
    for i in range(len(result)):
        if res == result[i]:
            ans.append(i)
    return ans

# 对于每一个非终结符号 求出其对应的所有存在于规则产生式右部的左部符号
def Get_dict(factor, result):
    vis = []
    res_dict = {}
    ans_dict = {}
    for i in range(len(result)):
        flag = False
        if vis.count(result[i]):continue
        vis.append(result[i])
        # res_dict[result[i]] = []
        tmp = {}
        for j in factor[i]:
            tmp[j] = []
            x = getlist(result, j)
            if x:
                for xx in x:tmp[j].append(xx)
                if result[i] not in res_dict.keys():res_dict[result[i]] = []
                res_dict[result[i]].append(j)
                flag = True
        # 递归求解
        while True:
            # 非终结符号k对应的产生式序号集合是l
            isneed = 0
            tmpp = {}
            for (k, l) in tmp.items():
                for m in l:
                    for n in factor[m]:
                        tmpp[n] = []
                        x = getlist(result, n)
                        if result[i] not in res_dict.keys():res_dict[result[i]] = []
                        if x and res_dict[result[i]].count(n)==0:
                            res_dict[result[i]].append(n)
                            flag = True
                            for xx in x:tmpp[n].append(xx)
                            isneed = 1
            if isneed==0:break
            tmp = dict(tmp, **tmpp)
            tmpp.clear()
        # print('tmp=', tmp)
        if flag:res_dict[result[i]] = list(set(res_dict[result[i]]))
        # print(res_dict)
        # 字典合并
        ans_dict = dict(ans_dict, **res_dict)
        res_dict.clear()
    return ans_dict

# 求出item前两项相同的所有key值
def Get_keyset1(graph, p0, p1):
    ans = []
    for (k, p) in graph.items():
        if p[0] == p0 and p[1] == p1:ans.append(k)
    return ans
# 求出item后两项相同的所有key值
def Get_keyset2(graph, p1, p2):
    ans = []
    for (k, p) in graph.items():
        if p[1] == p1 and p[2] == p2:ans.append(k)
    return ans
# 得到有向图
def Get_graph():
    factor, result = Get_G(PATH)
    animal = Get_animal(factor, result)
    r = Get_r(factor, result)
    rule = Get_rule(factor, result, r)
    ans_dict = Get_dict(factor, result)
    '''首先存储条件全部是非结论的结点信息。
       每个结点信息是长度为3的list，
       结点元素中，第一个元素是规则中非结论的条件集合，第二个元素是规则结果，第三个元素是规则中结论的条件集合'''
    graph = {}
    ind = 0
    for (res, fac) in rule.items():
        # graph[res] = [fac, None]
        graph[ind] = [fac, res, ['E']]
        ind += 1
    '''存储需要多次推理的结点信息'''
    for (key, val) in ans_dict.items():
        poslist = getlist(result, key)
        for index in poslist:
            # 这一条规则的条件集合 与 字典值的交集
            inter = list(set(factor[index]).intersection(set(val)))
            # 如果这条规则没有结论性条件，就直接建立新节点
            if len(inter) == 0:
                graph[ind] = [[factor[index]], result[index], ['E']]
                ind += 1
            # 如果包含结论性条件，就不断生成结点
            else:
                diff = list(set(factor[index]).difference(set(inter)))
                # 遍历交集，也就是遍历结论性条件
                for j in inter:
                    graph[ind] = [[diff], result[index], [j]]
                    ind += 1
    # 合并同类项
    tmp_graph = deepcopy(graph)
    for (k, p) in graph.items():
        keyset1 = Get_keyset1(tmp_graph, p[0], p[1])
        if len(keyset1) > 1:
            tmp_graph[keyset1[0]][2] = [tmp_graph[keyset1[0]][2]]
            for i in range(len(keyset1)):
                if i == 0:continue
                tmp_graph[keyset1[0]][2].append(tmp_graph[keyset1[i]][2])
                tmp_graph.pop(keyset1[i])
        keyset2 = Get_keyset2(tmp_graph, p[1], p[2])
        if len(keyset2) > 1:
            for i in range(len(keyset2)):
                if i == 0:continue
                tmp_graph[keyset2[0]][0].append(tmp_graph[keyset2[i]][0])
                tmp_graph.pop(keyset2[i])
    # 修改图结构
    graph.clear()
    for (k, p) in tmp_graph.items():
        graph[p[1]] = [p[0], p[2]]

    return graph

'''
ans_dict = 
{'有蹄类动物': ['哺乳动物'], '金钱豹': ['食肉动物', '哺乳动物'], 
'虎': ['食肉动物', '哺乳动物'], '长颈鹿': ['哺乳动物', '有蹄类动物'], 
'斑马': ['哺乳动物', '有蹄类动物'], '鸵鸟': ['鸟'], '企鹅': ['鸟'], 
'信天翁': ['鸟']}
'''

# grapth = Get_graph()
# for (k, p) in grapth.items():
#     print(k, ': ', p[0], '--', p[1],)
'''
value的第一项是并列关系，第二项（也就是后继）内容必须同时满足
graph =  
哺乳动物 :  [['有毛发'], ['有奶']] -- ['E']
鸟 :  [['有羽毛'], ['会下蛋', '会飞']] -- ['E']
食肉动物 :  [['吃肉'], ['有爪', '有犬齿', '眼盯前方']] -- ['E']
有蹄类动物 :  [['有蹄'], [['反刍动物']]] -- ['哺乳动物']
金钱豹 :  [['黄褐色', '暗斑点']] -- [['食肉动物'], ['哺乳动物']]
虎 :  [['黄褐色', '黑色条纹']] -- [['食肉动物'], ['哺乳动物']]
长颈鹿 :  [['长腿', '暗斑点', '长脖子']] -- ['有蹄类动物']
斑马 :  [['黑色条纹']] -- ['有蹄类动物']
鸵鸟 :  [['长腿', '长脖子', '不会飞', '黑白二色']] -- ['鸟']
企鹅 :  [['会游泳', '不会飞', '黑白二色']] -- ['鸟']
信天翁 :  [['善飞']] -- ['鸟']
'''
def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b
# 对于一个给定的结果res，返回所有可能的推理条件
def Get_condition(res, grapth):
    rules = []
    rule = []
    # 如果找不到该关键词，就返回空
    if res not in grapth.keys():return None
    # 否则开始推理
    rule.append(grapth[res][0])
    if grapth[res][1] == ['E']:
        for i in rule:
            rules.append(i)
            return rules
    nxt = []
    if len(grapth[res][1]) == 1 and grapth[res][1] != ['E']:
        nxt = [grapth[res][1]]
    else:nxt = grapth[res][1]
    while True:
        l = len(nxt)
        for i in range(l):
            if nxt[i] == ['E']:continue
            rule.append(grapth[nxt[i][0]][0])
            for j in grapth[nxt[i][0]][1]:
                nxt.append([j])
            nxt.remove(nxt[i])
        if nxt.count(['E']) == len(nxt):break
    '''
    rule= [[['暗斑点', '长腿', '长脖子']], [['有蹄'], [['反刍动物']]], [['有毛发'], ['有奶']]]
    rule= [[['黄褐色', '暗斑点']], [['吃肉'], ['有爪', '有犬齿', '眼盯前方']], [['有毛发'], ['有奶']]]
    '''
    # 条件组合
    for item in rule:
        if len(item) == 1:rules.append(item[0])
        else:
            for i in range(len(item)-1):
                for j in range(len(rules)):
                    rules.append(rules[j])
            j = 0
            for i in range(len(rules)):
                rules[i] = [rules[i]]
                rules[i].append(item[j])
                j += 1
                if j == len(item):
                    item.reverse()
                    j = 0
    # 条件拆分
    ans = []
    for r in rules:
        ans.append(flatten(r))
    return ans

def backward_inference(res, ):
    grapth = Get_graph()
    x=Get_condition(res, grapth)
    # for i in x:
    #     print(i)
    return x
# res = input('res=')
# backward_inference(res)
