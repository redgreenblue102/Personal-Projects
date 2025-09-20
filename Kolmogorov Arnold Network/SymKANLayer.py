import numpy as np 
import Kan_Bspline as bspline
import matplotlib.pyplot as plt
import sympy as sym

class SymKANLayer():
    def __init__(self, num_inputs, num_outputs,grid_size,grid_range,order):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.grid_size = grid_size
        self.grid_range = grid_range.copy()
        self.order = order
        self.funct = [[1 for i in range(num_inputs)] for j in range(num_outputs)]
        self.functparams
    def forwardpass():
        
        