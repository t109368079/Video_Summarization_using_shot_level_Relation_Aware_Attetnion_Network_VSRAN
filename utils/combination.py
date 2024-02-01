# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:44:40 2022

@author: Yuuki Misaki
"""
import copy 

def Combination(list_list):
    if len(list_list) > 2:
        sub_list = list_list.copy()
        first = sub_list[0]
        sub_list = sub_list[1:]
        return Combination([first, Combination(sub_list)])
    
    else:
        comb = []
        for x in list_list[0]:
            for y in list_list[1]:
                x_list = [x]
                if type(y) == type((1,2)):
                    y = list(y)
                    x_list += y
                else:
                    x_list.append(y)
                x_tuple = tuple(x_list)
                comb.append(x_tuple)
        return comb

def generate_comb(list_list, list_key):
    assert len(list_list) == len(list_key)
    comb_list = Combination(list_list)
    
    for i, l in enumerate(comb_list):
        temp = {}
        for j,k in enumerate(list_key):
            temp.update({k:l[j]})
        comb_list[i] = temp
    return comb_list

if __name__ == "__main__":
    l1 = [0,1]
    l2 = ['a','b']
    l3 = ['ア','イ']
    
    test = generate_comb([l1,l2,l3],['num','eng','jp'])
    print(test)

