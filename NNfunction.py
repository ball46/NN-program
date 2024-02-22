import numpy as np
import math


def sigmoid(v):
    """
    This is sigmoid function
    """
    s = 1 / (1 + math.exp(-v))
    return s


def Nout(x, w):
    """
    Sum of input,bias * weight(i)
    x=numpy.array() array of inputs and bias
    w=numpy.array() array of input weights and bias weight
    """
    o = sum(np.multiply(x, w))
    return o


def gradOut(e, y):
    """gradient of output node
    diff activation fuction is sigmoid
    y*(1-y)
        e is error of the node
    y is the output of the node"""
    g = e * y * (1 - y)
    return g


def gradH(y, sum):
    """gradient of hidden node
    diff activation fuction is sigmoid
    y*(1-y)
    y is the output of the node
    sum is sum of previous nodes* weight"""
    g = y * (1 - y) * sum
    return g


def deltaw(l, g, x):
    """Calculate the delta weight
    l is learning rate
    g is gradient of the node
    x is input of the node"""
    d = -l * g * x
    return d
