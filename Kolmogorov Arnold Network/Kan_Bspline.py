

import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
def spline(input, grid, jspline,order,extend=True):
    j=jspline+order
    if extend == False:
        gridspacing =grid[1]-grid[0]
        grid = np.append(np.array([grid[0]-(x+1)*gridspacing for x in reversed(range(order))]),grid)
        grid = np.append(grid,np.array([grid[-1] + (x+1)*gridspacing for x in range(order+1)]))
    return splinehelper(input,grid, j, order)

def splinehelper(input, grid, j,order):

    if(order == 0):
        if((grid[j]<=input) and (input < grid[j+1])):
            return 1
        else:
            return 0

    return ((input-grid[j])/(grid[j+order]-grid[j]))*splinehelper(input,grid,j,order-1) + \
        ((grid[j+order+1]-input)/(grid[j+order+1]-grid[j+1]))*splinehelper(input,grid, j+1,order-1)


def evaluatespline(input, coef, grid, order):
    return sum([coef[x+order]*(spline(input,grid,x,order)) for x in range(-order,len(grid)-2-2*order)])

def coefspline(input, output, grid, order):
    splineeval  = np.array([[spline(val,grid,j,order) for j in range(-order,len(grid))] for val in input])
    return scipy.linalg.lstsq(splineeval,output)[0]
def plotspline(input,coef,grid,order):
    sum = [evaluatespline(x,coef,grid,order) for x in input]
    for i in range(-order,len(grid)-2-2*order):
        plt.plot(input,coef[i+order]*np.array([spline(x,grid,i,order) for x in input.tolist()]),linestyle='--')
    plt.plot(input,sum,label="coef")
    plt.legend()
    return sum
def silu(input):
    if(input < -30):
        return -2.7e-12
    elif(input >20):
        return input
    return input/(1+math.exp(-input))
def siluderiv(input):
    if(input < -30):
        return -2.7e-12
    elif(input >20):
        return 1    
    return 1/(1+math.exp(-input))+input*math.exp(-input)/(1+math.exp(-input))**2
def main():
    '''xpoints = np.linspace(0,10,num=1001)
    gridsize = 11
    grid = np.linspace(0,10,num=gridsize)
    order = 5
    coefficients=np.random.rand(gridsize+order)
    print(coefficients)
    bsplines = [np.array([spline(x, grid, i,order)*coefficients[i+order] for x in xpoints]) for i in range(-order,gridsize)]
    for i in range(len(bsplines)):
        plt.plot(xpoints,bsplines[i])
    '''
    grid_1 = np.linspace(0,2,10)
    order_1 = 5
    input = np.linspace(0,2,num=100)
    output = [math.sin(k) for k in np.linspace(0,2,num=100)]
    coef=coefspline(input,output,grid_1,order_1)
    '''print(coef)
    sum = [evaluatespline(x,coef,grid_1,order_1) for x in input]
    for i in range(-order_1,len(grid_1)):
        plt.plot(input,coef[i+order_1]*np.array([spline(x,grid_1,i,order_1) for x in input.tolist()]))
    plt.plot(input,sum,label="coef")
    plt.plot(input,output,label="true")
    '''
    plotspline(input,coef,grid_1,order_1)
    plt.legend()
    plt.show()
#main()