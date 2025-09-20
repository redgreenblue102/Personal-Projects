import scipy
import matplotlib.pyplot as plt
import numpy as np
import math

def strong_wolfe(input,output,model,dir, a_low=0,a_high=1,a_def = 1e-3, c_1=0.0001,c_2 =0.9):
    y_0 = model.batchlossfunction(input,output)
    og_coef = model.getcoef()
    yder_0 = model.batchgradofloss(input,output)
    dyder_0 = np.sum([np.sum(np.multiply(dir[0][l],yder_0[0][l])) +\
                      np.sum(np.multiply(dir[1][l],yder_0[1][l])) for l in range(model.num_layers) ])
    while(a_high > a_low):
        a_cur = (a_low+a_high)/2
        newcoef = [[ np.add(og_coef[0][l],-dir[0][l]*a_cur)for l in range(model.num_layers)],\
                   [np.add(og_coef[1][l],-dir[1][l]*a_cur)for l in range(model.num_layers)] ]
        model.setcoef(newcoef)
        y_1 = model.batchlossfunction(input,output)
        yder_1 = model.batchgradofloss(input,output)
        dyder_1= np.sum([np.sum(np.multiply(dir[0][l],yder_1[0][l])) +\
                      np.sum(np.multiply(dir[1][l],yder_1[1][l])) for l in range(model.num_layers) ])
        if (y_1 <=y_0 -c_1*a_cur*dyder_0) \
            and (abs(dyder_1) <= abs(c_2*dyder_0)):
            model.setcoef(og_coef)
            print(a_cur)
            return a_cur
        if(y_1 <=y_0 -c_1*a_cur*dyder_0):
            a_low = a_cur+a_def
        else:
            a_high = a_cur -a_def
    model.setcoef(og_coef)
    print(a_cur)
    return a_cur
class Kan_LBFGS():
    def __init__(self,history_size):
        self.history_size = history_size
        self.prevvals = []
        self.gprev = 1
        self.paramsprev =1
    def lbfgsstep(self,g,params):
        i=0
        s = g- self.gprev
        y = params - self.paramsprev
        p = 1/sum(sum(sum(np.sum(np.multiply(y,s)))))
        if(len(self.prevvals) == self.history_size):
            self.prevvals.pop()
        self.prevvals.insert(0,[s,y,p])
        self.paramsprev = params
        self.gprev = g
        w=[]
        ug=g
        for s,y,p in self.prevvals:
            sug = sum(sum(sum(np.sum(np.multiply(s,ug)))))
            ssug = s*sug
            pssug = p*ssug
            w.insert(0,pssug)
            yg = np.multiply(y,ug)
            syg = s*yg
            ug = ug - p*syg
        i=0
        for s,y,p in reversed(self.prevvals):
            for j, x in enumerate(w[0:i]):
                sw = sum(sum(sum(np.sum(np.multiply(s,x)))))
                ysw = y*sw
                uw = x - p*ysw
                w[j] = uw
            
            sg = sum(sum(sum(np.sum(np.multiply(s,ug)))))
            ysg = y*sg
            ug = ug - p*ysg
            i+=1
        W=w[-1]
        for x in w[0:-1]:
            W = np.add(W,x)
        final = np.add(ug,W)
        return final
    def clear(self):
        self.prevvals = []
        self.gprev = 1
        self.paramsprev =1


            


