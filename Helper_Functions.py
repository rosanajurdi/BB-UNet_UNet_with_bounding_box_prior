'''
Created on Nov 27, 2019

@author: eljurros
'''
def threshold(array):
    array = (array > 0.91) * 1.0
    return array