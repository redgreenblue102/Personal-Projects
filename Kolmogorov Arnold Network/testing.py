from Kan_Model import Kan_Model
import Kan_Bspline as bspline
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

def trainsimple():
    model = Kan_Model([2,1,1],10,[-1,1],2)
    batchesin = [(np.random.rand(1000,2)*5) for i in range(1)]
    batchesout=[[math.sin(x+y) for x,y in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,iteration=200,wolfe=True,rangeextension=True,gridlock=True)
    file =open("simple.obj",'wb')
    pickle.dump(model,file)
    file.close()
def loadsimplemodel():
    file = open("simple.obj", 'rb') 
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig,inrange=[0,5])
    plt.show()
    return model
def trainsinmodel(it=1):
    model = Kan_Model([2,1,1],3,[-1,1],2)
    batchesin = [(np.random.rand(20,2)*2 -1) for i in range(20)]
    batchesout=[[function(x[0],x[1])for x in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,iteration=it,wolfe=True)
    file =open("model1.obj",'wb')
    pickle.dump(model,file)
    file.close()
def loadsinmodel():
    file = open("model1.obj", 'rb') 
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig)
    plt.show()
    return model
def function(x,y):
    return math.sin(x+y)

def traindiffmodel(model=None):
    batchesin = [(np.random.rand(40,2)*10) for i in range(40)]
    batchesout=[[math.sin(x+y)for x,y in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,learningrate=1,iteration=50,wolfe=True)
    file =open("model2.obj",'wb')
    pickle.dump(model,file)
    file.close()
def loaddiffmodel():
    file = open("model2.obj", 'rb') 
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig)
    return model
def trainmodel2(f,model=None):
    batchesin = [(np.random.rand(300,2)*5) for i in range(20)]
    batchesout=[[f(input) for input in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,iteration=20,wolfe=True,rangeextension=True,gridlock=True)
    file =open("model3.obj",'wb')
    pickle.dump(model,file)
    file.close()

def loadmodel2(model=None,ran =[-1,1]):
    file = open("model3.obj", 'rb') 
    #f = lambda x: ((math.sin(x[0]+x[1]) +  math.sqrt(x[0]+x[1])))**2
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig,inrange=ran)
    return model
def trainmodel3(f,model=None):
    batchesin = [(np.random.rand(100,2)*5-2.5) for i in range(10)]
    batchesout=[[f(input) for input in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,iteration=20,wolfe=True,rangeextension=True,gridlock=True,optimize =False)
    file =open("model4.obj",'wb')
    pickle.dump(model,file)
    file.close()

def loadmodel3(model=None,ran =[-1,1]):
    file = open("model4.obj", 'rb') 
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig,inrange=ran)
    return model
def trainmodel4(f,model=None):
    batchesin = [(np.random.rand(1000,2)*5-2.5) for i in range(1)]
    batchesout=[[f(input) for input in batch] for batch in batchesin]
    model.batchgradientdescent(batchesin,batchesout,iteration=100,wolfe=True,rangeextension=True,gridlock=True,optimize =False)
    file =open("model5.obj",'wb')
    pickle.dump(model,file)
    file.close()

def loadmodel4(model=None,ran =[-1,1]):
    file = open("model5.obj", 'rb') 
    model = pickle.load(file)
    file.close()
    fig = plt.figure()
    model.plotall(fig,inrange=ran)
    return model
'''trainsinmodel(it=20)
loadsinmodel()'''
'''model = Kan_Model([2,1,1],3,[-1,1],3)
traindiffmodel(model)
loaddiffmodel()
print(model.forwardpass([2,2]))
print(math.sin(4))'''
model = Kan_Model([2,1,1],10,[-1,1],2)
#f = lambda x: math.sin(2*(x[0]**3 -x[1]**2) + math.tanh(x[0] + math.cos(x[1])) )
f = lambda x: math.sin(x[0]**3 - x[1]**2)
trainmodel4(f,model)
model=loadmodel4(ran=[-2.5,2.5])
input =[1,1]
print(model.forwardpass(input))
print(f(input))
#trainsimple()
#model =loadsimplemodel()
plt.show()
