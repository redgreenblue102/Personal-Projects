import numpy as np
import sympy as sym  
import Kan_Bspline as bspline
import matplotlib.pyplot as plt
import sympy
class Kan_Layer():
    def __init__(self, num_inputs, num_outputs,grid_size,grid_range,order):
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.grid_size = grid_size
        self.order = order
        self.grid = np.linspace(grid_range[0],grid_range[1],num=grid_size+1)
        self.gridspacing = self.grid[1]-self.grid[0]
        self.grid = np.append(np.array([self.grid[0]-(x+1)*self.gridspacing for x in reversed(range(order))]),self.grid)
        self.grid= np.append(self.grid,np.array([self.grid[-1] + (x+1)*self.gridspacing for x in range(order+1)]))
        #self.coef = np.ones((grid_size+order,num_outputs,num_inputs))
        self.coef = np.random.rand(grid_size+order,num_outputs,num_inputs)*0.5-0.25
        #self.silucoef = np.ones((num_outputs,num_inputs))*0
        self.silucoef = np.random.rand(num_outputs,num_inputs)*(6/(num_inputs+num_outputs))**0.5
        self.grid_range=[grid_range[0],grid_range[1]]
        self.minmax =[grid_range[1],grid_range[0]]
        self.funcs = [[0 for i in range(num_inputs)] for j in range(num_outputs)]
        self.funcsparam = np.ones((num_outputs,num_inputs,4))

    def forward(self, x):
        return [sum([bspline.evaluatespline(x_1,self.coef[:,j,i],self.grid,self.order) + self.silucoef[j][i]*bspline.silu(x_1) for i,x_1 in enumerate(x)])\
                for j in range(self.num_outputs)]
    
    def gradLayer(self,x):
        grad= np.array([[[self.coef[k][i][j]*self.order*(bspline.spline(x[j],self.grid,k-self.order+1,self.order-1)/(self.grid[k+self.order] -self.grid[k] )\
                           - bspline.spline(x[j],self.grid,k-self.order+2,self.order-1)/(self.grid[k+self.order+1] -self.grid[k+1] )) \
                           if self.funcs[i][j] != 0 else 0 for j in range(self.num_inputs)]\
                          for i in range(self.num_outputs)] \
                          for k in range(self.grid_size+self.order)])
        return grad
    def gradsilu(self,x):
        grad = np.array([[self.silucoef[i][j]*bspline.siluderiv(x[j]) if self.funcs[i][j] != 0 else 0 for j in range(self.num_inputs)]\
                          for i in range(self.num_outputs)])
        return grad
    def gradsymbolic(self,input):
        x = sympy.symbols('x')
        grad = np.array([[self.funcsparam[2][i][j]*sympy.diff(self.funcs[i][j],x).subs(x,input).evalf()\
                           if self.funcs[i][j] != 0 else 0 for j in range(self.num_inputs)]\
                          for i in range(self.num_outputs)])
    def getlayercoef(self):
        return self.coef
    def getsilucoef(self):
        return self.silucoef
    def plotlayer(self,fig,layers,width,layer):
        input = np.linspace(self.minmax[0],self.minmax[1],num=1000)
        shape = str(layers) + str(self.num_inputs*self.num_outputs)
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                colspacing =(int(width/(self.num_inputs*self.num_outputs)))
                colindex = colspacing*(self.num_outputs*i+j)
                plt.subplot2grid((layers,width),(layer,colindex),colspan=colspacing).set_title(str(j)+str(i))
                sum =bspline.plotspline(input,self.coef[:,j,i],self.grid,self.order)
                total =np.add(np.array([self.silucoef[j][i]*bspline.silu(x) for x in input]),np.array(sum))
                plt.plot(input,[self.silucoef[j][i]*bspline.silu(x) for x in input],linestyle="--")
                plt.plot(input,total)
    def rangeextension(self,x,gridlock=False):
        if gridlock == False:
            if(x < self.grid_range[0]):
                    extend =int((self.grid_range[0]-x)/self.gridspacing)+1
                    self.grid_size=self.grid_size+extend
                    self.grid = np.append(np.array([self.grid[0]-(i+1)*self.gridspacing for i in reversed(range(extend))]),self.grid)
                    self.coef = np.append(np.random.rand(extend,self.num_outputs,self.num_inputs)*0.2-0.1,self.coef,axis=0)
                    self.grid_range[0] =self.grid_range[0] -extend*self.gridspacing
            elif(x > self.grid_range[1]):
                extend =int((x-self.grid_range[1])/self.gridspacing)+1
                self.grid_size=self.grid_size+extend
                self.grid = np.append(self.grid,np.array([self.grid[-1]+(i+1)*self.gridspacing for i in range(extend)]))
                self.coef = np.append(self.coef,np.random.rand(extend,self.num_outputs,self.num_inputs)*0.2-0.1,axis=0)
                self.grid_range[1] =self.grid_range[1]+extend*self.gridspacing
        else:
            if(x < self.grid_range[0]):
                self.grid = np.linspace(x,self.grid_range[1],num=self.grid_size+1)
                self.gridspacing = self.grid[1]-self.grid[0]
                self.grid = np.append(np.array([self.grid[0]-(i+1)*self.gridspacing for i in reversed(range(self.order))]),self.grid)
                self.grid= np.append(self.grid,np.array([self.grid[-1] + (i+1)*self.gridspacing for i in range(self.order+1)]))
                self.grid_range[0] =x
            elif(x > self.grid_range[1]):
                self.grid = np.linspace(self.grid_range[0],x,num=self.grid_size+1)
                self.gridspacing = self.grid[1]-self.grid[0]
                self.grid = np.append(np.array([self.grid[0]-(i+1)*self.gridspacing for i in reversed(range(self.order))]),self.grid)
                self.grid= np.append(self.grid,np.array([self.grid[-1] + (i+1)*self.gridspacing for i in range(self.order+1)]))
                self.grid_range[1] =x
    def fix_sym(self,input,output,func="0"):
        self.funcs[output][input] = sympy.sympify(func)

def main():
    layer = Kan_Layer(2,2,2,[0,2],2)
    print(layer.forward([1,1]))
    print(layer.gradLayer([1.5,0.5]))
    print(layer.gradsilu([1.5,0.5]))
    print("coefficients here")
    fig=plt.figure()
    layer.plotlayer(fig,1,4,0)
    
    plt.show()
#main()
