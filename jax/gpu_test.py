
from jax import device_put
from numba import jit, cuda
import numpy as np

# to measure exec time
from timeit import default_timer as timer   
n = 100000000

# normal function to run on cpu
def func(a):                                
    for i in range(n):
        a[i]+= 1     
 
# function optimized to run on gpu 
@jit(target_backend='cuda')                         
def func2(a):
    for i in range(n):
        a[i]+= 1
if __name__=="__main__":
    a = np.ones(n, dtype = np.float64)

    start = timer()
    func(a)
    print("without GPU:", timer()-start)    
    
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)