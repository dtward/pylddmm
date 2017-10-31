# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:38:23 2017

@author: dtward
"""

import numpy as np
from multiprocessing import Pool

nBig = 5
nSmall = 10
nWorkers = 2
def dummy_func(x):
    a = x*0
    for i in range(1000):
        a += x
    return a
    
'''
for i in range(nBig):
    print('Starting big loop {}'.format(i))
    p = Pool(processes = nWorkers)
    data = [np.random.rand(1000000) for _ in range(nSmall)]
    out = p.map(dummy_func,data)    
    p.close()
    p.join()
    # here I would do some proccessing on the output
'''
# on iteration 0 memory is 1.4% x2 processes
# on iteration 1 memory is 5.3% x2 processes
# on iteration 2 memory is 6.4% x2 processes
# on iteration 3 memory is 6.7% x2 processes
# on iteration 4 memory is 7.5% x2 processes
# on iteration 5 memory is 6.5% x2 processes
        
'''
# on the other hand
for i in range(nBig):
    print('Starting big loop {}'.format(i))    
    #data = [np.random.rand(1000000) for _ in range(nSmall)]
    out = []
    for i in range(nSmall):
        data_i = np.random.rand(1000000)
        out.append(dummy_func(data_i))
    # here I would do some processing on the output
'''

for i in range(nBig):
    print('Starting big loop {}'.format(i))    
    data = [np.random.rand(1000000) for _ in range(nSmall)]
    out = []
    for i in range(nSmall):
        out.append(dummy_func(data[i]))
    # here I would do some processing on the output