# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 08:56:54 2017

@author: dtward
"""

from multiprocessing import Pool
from multiprocessing import Process
import os

def f(x):
    return x*x

data = [1,2,3,4,5]
n = len(data)
p = Pool(n)
out = p.map(f,data)
print(out)

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):  # only available on Unix
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()

def f(name):
    info('function f')
    print 'hello', name
    
info('main line')
p = Process(target=f, args=('bob',))
p.start()
p.join()


def do_nothing(x,test=1):
    return x
def multiply(x,y):
    x = do_nothing(x)
    return x*y
def multiply_tuple(x):
    return multiply(*x)
    
p = Pool(2)
out = p.map(multiply_tuple,((5.0,6.0),(7,8),(8,9)))
print(out)