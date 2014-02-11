"""
gpo.py  - optimization using Gaussian process regression. 
Marcus Frean. 

usage: python gpo.py [datafile]

Reads a initial set of input vectors (x) from the file. Output (y)
values for these (and any future) x are given by one of the functions
in testFunctions.py

The task is then to choose where to sample next.
Each successive sample (x,y) taken in this way is added to the dataset.
"""

import sys
import pylab
from numpy import *
import numpy.random as rng
from scipy.optimize import fmin, fmin_cg, check_grad
#from scipy.io import write_array, read_array
#from scipy.io import read_array
from scipy.special import erf, erfc
import testFunctions, gp



#-------------------------------------------------------------------------

def takeSample(x):
    y = testFunctions.funcA(x)
    # print x,' --->  ',y
    return y


def calcNegEI(x, EIArgs, MarcusCorrection=False):
    # It will help to make x into a row vector (1 by D matrix) first....
    D = len(ravel(x))
    x = x.reshape((1,D))

    (thetaArgs,theta,best_so_far) = EIArgs 
    (X,Y,hyperprior) = thetaArgs
    mu,sig2 = gp.calcGPPrediction(theta,thetaArgs,x)
    sig = sqrt(sig2)
    u = (mu - best_so_far)/sig
    Phi = 0.5*erf(u/sqrt(2)) + 0.5
    phi = 1.0/sqrt(2*pi) * exp(-pow(u,2)/2.0)

    if MarcusCorrection:
        (n,D,K,Q,L,alpha) = gp.doCholesky(theta,thetaArgs)
        # alpha = C^{-1}_N k_star
        # so transpose(alpha) = k_\star^T C^{-1}_N 
        kstar = gp.calcCovariance(X,x,theta)
        kappa = gp.calcCovariance(x,x,theta)
        meanPred = dot(transpose(alpha),kstar)
        V = linalg.solve(L,kstar)
        verticalScale = exp(theta[D])
        varPred = ravel( verticalScale - diag(dot(transpose(V), V)) )
        m = 1.0/varPred
        mVec = -m * alpha
        M = 0 # HERE IS WHERE MICHAEL IS UP TO!!!!!!!!!!!!!!!!!!!!

    EI = sig*(u*Phi  + phi)

    return -EI  # minus since fmin_cg looks for a MINIMUM


def calcNegGradEI(x, EIArgs):
    """
    calc the negative of the gradient of the expected improvement.
    I'm following the derivation given in Phillip Boyle's thesis.
    """
    # It will help to make x into a row vector (1 by D matrix) first....
    D = len(ravel(x))
    x = x.reshape((1,D))

    (thetaArgs,theta,besty) = EIArgs 
    (X,Y,hyperprior) = thetaArgs
    mu,sig2 = gp.calcGPPrediction(theta,thetaArgs,x)
    #print 'mu is ',mu,', sig2 is ',sig2
    sig = sqrt(sig2)
    u = (mu - besty)/sig
    Phi = 0.5*erf(u/sqrt(2)) + 0.5
    phi = 1.0/sqrt(2*pi) * exp(-pow(u,2)/2.0)
    #print 'u is ',u,', Phi is ',Phi, ', phi is ',phi

    # some replication of calcGPPrediction in gp.py here, to get derivs.
    (n,D,K,Q,L,alpha) = gp.doCholesky(theta,thetaArgs)
    k = gp.calcCovariance(X,x,theta)
    invQ = linalg.solve(transpose(L), linalg.solve(L,eye(n)))  

    # first, the dk/dx vector:
    dkdx = zeros((D,n))
    for d in range(D):
        lengthScale = exp(theta[d])
        dkdx[d,:] = transpose(k) * 2*lengthScale*(X[:,d]-x[0,d]) 
        # note that is elt-wise multiplication.
    #print 'dkdx is ',transpose(dkdx)
    # then dsdx
    dsdx = -(dot(dkdx, dot(invQ,k)))/sig
    #print 'dsdx is ',transpose(dsdx)
    # then dudx
    tmp = (dot(dkdx,dot(invQ,Y)).reshape((D,1)))
    dudx = (tmp - u*dsdx) / sig
    #print 'dudx is ',transpose(dudx)

    # now put them all together to get EIderiv:
    EIderiv = (u*Phi+phi)*dsdx  +  sig*Phi*dudx
    EIderiv = EIderiv.reshape((D,))
    #print 'EIderiv is ',transpose(EIderiv)

    return -EIderiv/2  # minus because fmin_cg finds MINIMUM



def simulatedAnnealingEI(eiArgs,basex): 
    """
    Starting at the best x so far, run simulated annealing on the EI
    surface to choose a new point to sample at.
    But there's still something wrong with this code!!!
    SA returns silly results sometimes.
    """
    epsilon = 5.0  # max distance from the best x so far.
    nItnsSA = 1000
    x = basex # start at the best x sampled at so far.
    EI = -calcNegEI(x,eiArgs)
    numAccepted = 0
    for t in range(nItnsSA):
        temperature = 0.1*(nItnsSA-t)/(1.0*nItnsSA) # from 1 to 0.
        while 1:
            proposalx = x + rng.normal((D,))
            diff = proposalx - basex # only consider nearby proposals
            if sqrt(dot(diff,diff)) < epsilon:
                break
        proposalEI = -calcNegEI(proposalx,eiArgs)
        acceptProb = min(1.0, exp((proposalEI-EI)/temperature))
        if rng.random() < acceptProb:
            x,EI = proposalx,proposalEI
            numAccepted += 1
    print 'ratio accepted = ',numAccepted*1.0/nItnsSA
    return (x,EI)


def conjugateGradientsEI(eiArgs):
    #g1 = calcNegGradEI(startx, EIArgs) # gradient found by analytic method
    # Following is test that grad calc is correct, using check_grad.
    #testGradientCalc = check_grad(calcNegEI, calcNegGradEI, startx, eiArgs)
    #if testGradientCalc > 0.01:
    #    sys.exit("calcNegGradEI does not give correct grad of calcNegEI!")

    numRestartsCG = 10 # number of times CG is run from random restarts.
    EI = -1000000.0
    for t in range(numRestartsCG):  # restarts, to avoid local min
        startx = copy(X[rng.randint(0,n),:]) # start near any previous sample
        startx = startx + 1.0*rng.normal(shape(startx)) # since gradient ~ 0 at a data point.
        # call conjugate gradients method.
        xopt = fmin_cg(calcNegEI, startx, calcNegGradEI, [eiArgs], gtol=1e-4,disp=0)
        EIopt = -calcNegEI(xopt, eiArgs)
        if EIopt > EI:
            x, EI = xopt, EIopt
    return (x,EI)

#-----------------------------------------------------------


if __name__ == '__main__': 



    D = 1 # dimensionality of search space
    yoffset = 0.0 # to keep track of shifts up and down to y-axis
    maxNumSamples = 6  # maximum total number of samples
    #rng.seed(1101) # comment this out to get a random seed.
    n = 2 # number of initial inputs
    xlim = 1 # we will only consider solutions where abs(x) < xlim

    (theta, hyperprior) = gp.setAndSampleHyperprior(D) 

    X = zeros((n,D))
    for i in range(n):
        X[i,:] = rng.normal(D)
    Y = zeros((n,))
    for i in range(n):
        Y[i] = takeSample(X[i,:])

    currentEI = 1.0
    counter = 1
    while currentEI>0.0001 and n<maxNumSamples:
        # set the arguments the optimizers need.
        thetaArgs = (X,Y,hyperprior) # fmin_cg needs these all in one box.
        # find MAP hyperparameters, given all available data.
        theta = fmin_cg(gp.calcNegLogPosterior, theta, gp.calcNegGradLogPosterior, [thetaArgs], gtol=1e-2,disp=0)
        besty = Y.max()
        EIArgs = (thetaArgs,theta,besty) # fmin_cg needs these.
        
            
        # find an X (newX) that maximizes the EI measure.
        EI_max = -1000000
        for x_start in X: # try starting from each data point in turn
            x_fmin = fmin(calcNegEI, x_start, [EIArgs],disp=0)
            EI_fmin = -calcNegEI(x_fmin, EIArgs)
            if EI_fmin > EI_max:
                EI_max = EI_fmin
                nextx = x_fmin
        # Some other variations to try if you're unhappy with fmin's performance.
        #x_CG, EI_CG = conjugateGradientsEI(EIArgs) # use Conj. Grad.
        #x_SA, EI_SA = simulatedAnnealingEI(EIArgs,x_start) # use Sim. Anneal
        #print 'x: EXHAUSTIVE ',x_EX,'\t CG ',x_CG,'\t SA ', x_SA,'\t fmin ', x_fmin

        currentEI = EI_fmin
    
        # Store x (i.e. increase X by one row), and store y in Y.
        l = list(X)
        l.append(nextx)
        n=len(l)
        X = asarray(l)
        l = list(Y+yoffset) # the +yoffset reconverts back to true Y
        y = takeSample(nextx) # takes a sample at x
        l.append(y)
        Y = asarray(l)

        # Hack alert: moving Y up/down semi-arbitrarily!!!
        yoffset = Y.mean()
        Y = Y-yoffset

        


    thetaArgs = (X,Y,hyperprior)
    theta = fmin_cg(gp.calcNegLogPosterior, theta, gp.calcNegGradLogPosterior, [thetaArgs], gtol=1e-2,disp=0)
    besty = Y.max()
    EIArgs = (thetaArgs,theta,besty)


    for sample in range(len(ravel(Y))):
        print X[sample,:], '\t --> \t',Y[sample]

    # Write out the samples to a file.
    data = zeros((shape(X)[0], shape(X)[1] + 1))
    data[:,0:-1] = X    
    data[:,-1] = Y
    #write_array("outputGPOdata.txt", data)   deprecated: needs fixing...

