import numpy as np
from Kan_Layer import Kan_Layer
import Kan_Bspline as bspline
import math
import matplotlib.pyplot as plt
import Kan_LBFGS as lb
from Kan_LBFGS import Kan_LBFGS
import multiprocessing as mp
from multiprocessing import shared_memory
class Kan_Model:
    def __init__(self,width,grid_size,grid_range,order):
        self.grid_size = grid_size
        self.order =order
        self.width = width
        self.grid_range = grid_range
        self.layers = [Kan_Layer(width[i],width[i+1],grid_size,grid_range,order) for i in range(len(width)-1)]
        self.num_layers = len(self.layers)

    def grad(self,forwardevals):
        return [[self.layers[i].gradLayer(forwardevals[i]) ,\
                self.layers[i].gradsilu(forwardevals[i])] for i in range(self.num_layers)]
    def gradofloss(self,y,forwardevals):
        gradients=self.grad(forwardevals)
        totalgrad = []
        passing = np.diag(np.subtract(forwardevals[self.num_layers],np.array(y)))
        silugrad = []
        sym_grad = []
        for l in reversed(range(self.num_layers)):
            gradients[l] = np.add(np.sum(gradients[l][0],0),gradients[l][1])
            grad = np.zeros((self.layers[l].grid_size+self.layers[l].order,self.layers[l].num_outputs,self.layers[l].num_inputs))
            silugradlayer = np.zeros((self.layers[l].num_outputs,self.layers[l].num_inputs))
            for i in range(self.layers[l].num_outputs):
                for j in range(self.layers[l].num_inputs):
                    loc = forwardevals[l][j]
                    silugradlayer[i][j] = np.sum(bspline.silu(loc) * passing[i,:])
                    #silugradlayer[i][j] = 0
                    for k in range(self.layers[l].grid_size+self.layers[l].order):
                        grad[k][i][j] =np.sum(bspline.spline(loc,self.layers[l].grid\
                                                       ,k-self.layers[l].order,self.layers[l].order)*passing[i,:])
            totalgrad.insert(0,grad)
            silugrad.insert(0,silugradlayer)
            passing = np.dot(np.transpose(gradients[l]),passing)
        return [totalgrad,silugrad]
    def create_shared_memory_nparray(data):
        arrshape = (2,)
        datatype=np.obj2sctype
        d_size = np.dtype(datatype).itemsize * np.prod(arrshape)
        
        shm = shared_memory.SharedMemory(create=True, size=d_size, name='totalgrad')
        # numpy array on shared memory buffer
        dst = np.ndarray(shape=arrshape, dtype=datatype, buffer=shm.buf)
        dst[:] = data[:]
        return shm


    def release_shared(name):
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()  # Free and release the shared memory block
    def batchgradofloss(self,batch_x,batch_y,cores=1):

        totalgrad= np.array([[np.zeros((x.grid_size+x.order,x.num_outputs,x.num_inputs)) for x in self.layers],\
                    [np.zeros((x.num_outputs,x.num_inputs)) for x in self.layers]],dtype=object)
        size = len(batch_x)
        #pool = mp.Pool(mp.cpu_count())
        #pool.starmap(self.processfunc,zip([x for x in batch_x],[y for y in batch_y],[totalgrad for i in range(size)]))
        for x,y in zip(batch_x,batch_y):
            temp = self.gradofloss(y,self.forwardpass(x))
            for l in range(self.num_layers):
                np.add(temp[0][l],totalgrad[0][l], out = totalgrad[0][l] )
                np.add(temp[1][l],totalgrad[1][l],out =totalgrad[1][l])  
        for l in range(self.num_layers):
            np.divide(totalgrad[0][l],size,out = totalgrad[0][l] )
            np.divide(totalgrad[1][l],size,out =totalgrad[1][l] )
        return totalgrad
    def processfunc(self,x,y,totalgrad):
        temp = self.gradofloss(y,self.forwardpass(x))
        for l in range(self.num_layers):
            totalgrad[0][l] = np.add(np.array(temp[0][l]),totalgrad[0][l])
            totalgrad[1][l] = np.add(np.array(temp[1][l]),totalgrad[1][l])    

    def lossfunction(self,x,y):
        residual=np.subtract(np.array(y),np.array(self.forwardpass(x,self.num_layers)[self.num_layers]))
        loss = np.dot(residual,residual)/2
        return loss
    
    def batchlossfunction(self,batch_x,batch_y):
        sum = 0
        for x,y in zip(batch_x,batch_y):
            sum += self.lossfunction(x,y)
        sum = sum/len(batch_x)
        return sum

    def gradientdescent(self,batchx,batchy,learningrate=1,iteration=100,wolfe =False,rangeextension=True,gridlock=False,optimize=None):
        if (optimize != None):
            optim=optimize
            if rangeextension== True:
                self.rangeupdate(batchx,gridlock=gridlock)
            delta = learningrate
            grad=self.batchgradofloss(batchx,batchy)
            optim.gprev = grad.copy()
            optim.paramsprev = np.array([[np.copy(z.coef) for z in self.layers],[np.copy(z.silucoef) for z in self.layers ]],dtype=object)
            if wolfe == True:
                delta= lb.strong_wolfe(batchx,batchy,self,grad)
            for l,z in enumerate(self.layers):
                z.coef = np.subtract(z.coef,np.multiply(delta,np.array(grad[0][l])))  
                z.silucoef = np.subtract(z.silucoef,np.multiply(delta,np.array(grad[1][l])))       
        for i in range(iteration):
            loss=self.batchlossfunction(batchx,batchy)
            print(f"{i}iter{loss}")
            if rangeextension== True:
                self.rangeupdate(batchx,gridlock=gridlock)
            delta = learningrate
            grad=self.batchgradofloss(batchx,batchy)
            if optimize != None:
                params = np.array([[np.copy(z.coef) for z in self.layers],[np.copy(z.silucoef) for z in self.layers ]],dtype=object)
                dir = optim.lbfgsstep(np.copy(grad),params)
            else:
                dir = grad
            if wolfe == True:
                delta= lb.strong_wolfe(batchx,batchy,self,dir)
            for l,z in enumerate(self.layers):
                z.coef = np.subtract(z.coef,np.multiply(delta,np.array(dir[0][l])))  
                z.silucoef = np.subtract(z.silucoef,np.multiply(delta,np.array(dir[1][l])))
                #print(f"layer:{l}")
                #print(z.silucoef)  
        print(self.batchlossfunction(batchx,batchy))
    def batchgradientdescent(self,batchesx,batchesy,learningrate=1,iteration=100,wolfe =False,rangeextension =True,gridlock=False,optimize =False):
        b=0
        if optimize == True:
            optimizer = Kan_LBFGS(10)
        else:
            optimizer = None
        for x,y in zip(batchesx,batchesy):
            print(f"batch{b}")
            self.gradientdescent(x,y,learningrate=learningrate,iteration=iteration,wolfe=wolfe,rangeextension=rangeextension,gridlock=gridlock,optimize=optimizer)
            b += 1
    def forwardpass(self,x,l=None):
        if(l == None):
            l=self.num_layers
        layereval = []
        layereval.append(x)
        for i in range(l):
            x = self.layers[i].forward(x)
            layereval.append(x)

        return layereval
    def rangeupdate(self,batchx,gridlock=False):
        maxval=[layer.grid_range[1] for layer in self.layers]
        minval = [layer.grid_range[0] for layer in self.layers]
        for x in batchx:
            forward = self.forwardpass(x,self.num_layers)
            for l in range(self.num_layers):
                maxtemp = max(forward[l])
                if maxtemp > maxval[l]:
                    maxval[l] = maxtemp
                mintemp = min(forward[l])
                if mintemp < minval[l]:
                    minval[l] = mintemp
        for l in range(self.num_layers):
            self.layers[l].rangeextension(maxval[l],gridlock=gridlock)
            self.layers[l].rangeextension(minval[l],gridlock=gridlock)
    def getcoef(self):
        array=[]
        siluarray = []
        for i,x in enumerate(self.layers):
            array.append(x.getlayercoef())
            siluarray.append(x.getsilucoef())
        return np.array([array,siluarray],dtype=object)
    def plotall(self,fig,inrange=[-1,1]):
        
        input = np.random.rand(500,self.width[0])*(inrange[1]-inrange[0])+inrange[0]
        for x in input:
            temp=self.forwardpass(x)
            for l in range(self.num_layers):
                m = max(temp[l])
                if  m > self.layers[l].minmax[1]:
                    self.layers[l].minmax[1] = m
                m = min(temp[l])
                if m < self.layers[l].minmax[0]:
                    self.layers[l].minmax[0] = m
        width = max([self.width[i]*self.width[i+1] for i in range(len(self.width)-1)])

        for i,x in enumerate(reversed(self.layers)):
            x.plotlayer(fig,self.num_layers,width,i)
    def setcoef(self,coef):
        for i,x in enumerate(self.layers):
            x.coef = np.array(coef[0][i])
            x.silucoef = np.array(coef[1][i])
    def fix_symbolic(self,layer,output,input,func = "0"):
        self.layers[layer].fix_sym(output,input,func=func)
  

def main():
    num_input = 1
    model = Kan_Model([num_input,2],1,[-10,10],2)
    batchin = (np.random.rand(30,num_input)*10 -5).tolist()
    #batchout=[function(x,y) for x,y in batchin]
    model.gradofloss([1],[1,1])
    #model.gradientdescent(batchin,batchout)
    #print(model.forwardpass([0.5,0.9]))
    #print(model.forwardpass([-0.5,-0.5]))
    #print(parabola(0.5,0.9))
    #print(model.lossfunction([0.5,0.9],parabola(0.5,0.9)))
    print(model.forwardpass([1]))
    print(model.forwardpass([1]))
    fig = plt.figure()
    model.plotall(fig)
    plt.show()
def function(x,y):
    return x+y