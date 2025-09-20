import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import sympy

z=2
x, y = sympy.symbols('x y')
expr = sympy.cos(x) +1
expression = sympy.sympify("cos(u)**2")
print(sympy.diff(expression,u))
print(sympy.diff(expr,x))
print(expr.subs(x,z))