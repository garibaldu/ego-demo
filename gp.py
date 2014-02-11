"""
gp.py  - a Gaussian process regression. Marcus Frean. 

usage: python gp.py datafile

Gaussian process regression. Reads a datafile consisting of a matrix
in which each row is a training item. The final column in the target
and preceding columns are the input.

Hyperparameters theta control a squared exponential covariance
function, and are chosen from Gaussian distributions. These theta are
exponentiated to give the length scales in each input dimension, the
overall vertical scale of the covariance function (i.e. the max), and
the amount of noise assumed to be corrupting the observed target
values. That is:
length scale in 1st dim of input =  exp(theta[0])
length scale in 2nd dim of input =  exp(theta[1])
  :
  :
length scale in n-th dim of inpt =  exp(theta[D-1])
Vertical scale of covariance fn  =  exp(theta[D])
Assumed variance of target noise =  exp(theta[D+1])
NOTE THE EXPONENTIATION OF THETA (ie: theta is in logspace), which allows 
us to put simple Gaussian priors on theta.

"""


import sys
import pylab
from numpy import *
import numpy.random as rng
from scipy.optimize import fmin_cg, check_grad
#from scipy.io import write_array


def calcCovariance(X1, X2, theta):
    """
    This works for either making a square cov matrix from data
    via calcCovariance(X,X,theta), OR making a vector of
    covariances of some test data (against train) via
    calcCovariance(X, Xtest, theta), where Xtest is a matrix of
    test data with same form as X (i.e. one pattern = one row).
    """
    (n1,D)  = shape(X1)
    if len(shape(X2)) != 2:
        print 'warning in gp.calcCovariance: shape(X2) is ',shape(X2)
        L = len(X2)
        X2 = X2.reshape((1,D))
    (n2,D2) = shape(X2)
    if (D2 != D):
        print 'X1 and X2 should have same number of cols ',D,D2
        print 'X1.shape: ',X1.shape
        print 'X2.shape: ',X2.shape
        sys.exit("calcCovariance in gp.py: oops, dimensions are wrong!");
    S = zeros((n1,n2))
    for d in range(D):
        A1 = transpose([X1[:,d]]) * ones((n1,n2))
        A2 = [X2[:,d]] * ones((n1,n2))
        S = S + pow(A1-A2,2) * exp(theta[d])
    verticalScale = exp(theta[D])
    K = verticalScale * exp(-0.5 * S)
    return K


def calcCovariance_superposition(X1, X2, theta_A, theta_B):
    """ Copies calcCovariance(), but has TWO covariance functions added together.
        So far this is done very crudely and both have to have same number / format hyperparameters.
    """
    (n1,D)  = shape(X1)
    if len(shape(X2)) != 2:
        print 'warning: shape(X2) is ',shape(X2)
        L = len(X2)
        X2 = X2.reshape((1,D))
    (n2,D2) = shape(X2)
    if (D2 != D):
        print 'X1 and X2 should have same number of cols ',D,D2
        print 'X1: ',X1
        print 'X2: ',X2
        sys.exit("oops!")

    S_A = zeros((n1,n2))
    S_B = zeros((n1,n2))
    for d in range(D):
        A1 = transpose([X1[:,d]]) * ones((n1,n2))
        A2 = [X2[:,d]] * ones((n1,n2))
        sqd_diff = pow(A1-A2,2)# use for squared exponential covariance fns.
        abs_diff = abs(A1-A2)  # use if you want Brownian motion instead.
        S_A = S_A + sqd_diff * exp(theta_A[d])
        S_B = S_B + sqd_diff * exp(theta_B[d])
    verticalScale_A = exp(theta_A[D])
    verticalScale_B = exp(theta_B[D])

    K = verticalScale_A * exp(-0.5 * S_A)  + verticalScale_B * exp(-0.5 * S_B)
    return K


def setAndSampleHyperprior(D):
    # SET HYPERPRIORS FOR ALL PARAMETERS
    hyperLogInvLengths  = (log(10.0), 5.0); # prior for inverse length scales
    hyperLogVertical = (log(1.0), 5.0);# prior for vertical scale
    hyperLogNoiseVar = (log(0.001),  1.0);# prior for noise variance
    # collect them under one name
    hyperprior = (hyperLogInvLengths,hyperLogVertical,hyperLogNoiseVar);

    # INITIAL HYPERPARAMETERS, CHOSEN FROM THE HYPERPRIOR
    logInvLengths  = rng.normal(hyperLogInvLengths[0],hyperLogInvLengths[1],(D)) 
    logVertical = rng.normal(hyperLogVertical[0],hyperLogVertical[1])
    logNoiseVar = rng.normal(hyperLogNoiseVar[0],hyperLogNoiseVar[1])
    # collect them as one vector
    theta = list(logInvLengths)  # indexed from 0..D-1
    theta.append(logVertical) # indexed at D
    theta.append(logNoiseVar) # indexed at D+1
    return (theta, hyperprior)

def doCholesky(theta, args):
    # These same steps are necessary before calculating predictions,
    # logPosterior and its gradients, so I'm collecting them here.
    (X,y,hyperprior) = args
    (n,D) = shape(X)
    noiseVar  = exp(theta[D+1])
    K = calcCovariance(X,X,theta)
    Q = K + noiseVar*eye(n)  # we're adding in the observation noise here.
    L = linalg.cholesky(Q)   # Cholesky factorization kicks ass.
    alpha = linalg.solve(transpose(L) , linalg.solve(L,y))
    return (n,D,K,Q,L,alpha)


def calcGPPrediction(theta,args,Xtest):
    """
    Make a prediction about the output given input Xtest (can be several).
    """
    (n,D,K,Q,L,alpha) = doCholesky(theta,args)

    (X,y,hyperprior) = args
    kstar = calcCovariance(X,Xtest,theta)
    meanPred = dot(transpose(kstar),alpha)
    V = linalg.solve(L,kstar)
    verticalScale = exp(theta[D])
    varPred = ravel( verticalScale - diag(dot(transpose(V), V)) )
    return (meanPred,varPred)


def calcNegLogPosterior(theta, args):
    (n,D,K,Q,L,alpha) = doCholesky(theta,args)
    (X,y,hyperprior) = args
    logL = -0.5*dot(transpose(y),alpha)  - sum(log(diag(L))) - 0.5*n*log(2*pi);

    # hang on: haven't included the log prob of hyperparams, under hyperprior.
    (logProbTheta, gradLogProbTheta) = calcLogProbTheta(theta,hyperprior)
    
    return -(logL + sum(logProbTheta))  # minus since fmin_cg looks for a MINIMUM

def calcLogProbTheta(theta, hyperprior):
    """
    Returns the log prob of each component of theta under the
    hyperprior, and its gradient.
    """
    (hyperLogInvLengths,hyperLogVertical,hyperLogNoiseVar) = hyperprior
    logP = zeros(shape(theta))
    gradLogP = zeros(shape(theta))
    D = len(theta)-2
    for d in range(D):
        logP[d] = -pow(hyperLogInvLengths[0] - theta[d],2.0)/(2*hyperLogInvLengths[1])
        logP[d] -= 0.5*log(2*pi*hyperLogInvLengths[1])
        gradLogP[d] = (hyperLogInvLengths[0] - theta[d])/hyperLogInvLengths[1]
    logP[D] = -pow(hyperLogVertical[0] - theta[D],2.0)/(2*hyperLogVertical[1])
    logP[D] -= 0.5*log(2*pi*hyperLogVertical[1])
    gradLogP[D] = (hyperLogVertical[0] - theta[D])/hyperLogVertical[1]
    logP[D+1] = -pow(hyperLogNoiseVar[0] - theta[D+1],2.0)/(2*hyperLogNoiseVar[1])
    logP[D+1] -= 0.5*log(2*pi*hyperLogNoiseVar[1])
    gradLogP[D+1] = (hyperLogNoiseVar[0] - theta[D+1])/hyperLogNoiseVar[1]
    return(logP,gradLogP)


def calcNegGradLogPosterior(theta, args):
    """
    calc the negative gradient of the log likelihood of the data.
    """
    (n,D,K,Q,L,alpha) = doCholesky(theta,args)
    (X,y,hyperprior) = args
    invQ = linalg.solve(transpose(L),  linalg.solve(L,eye(n)) )  
    # yep, I've checked and this seems to be inv(Q). Tick.

    dLogL = zeros(shape(theta))
    invQt = dot(invQ,y)
    # need matrix of derivatives, dK/dtheta_i, which depends on the cov fn form.
    for d in range(D):
        lengthScale = exp(theta[d])
        B = [X[:,d]] * ones((n,n))
        V = pow(B-transpose(B),2) * K
        term1 = dot(transpose(invQt), dot(V,invQt))
        term2 = sum(sum(invQ*V))
        dLogL[d] = -lengthScale * 0.25 * (term1 - term2)
    dLogL[D] = -0.5*(sum(sum(invQ*K)) - dot(transpose(invQt), dot(K,invQt)))
    dLogL[D+1] = -0.5*exp(theta[D+1])*(trace(invQ) - dot(transpose(invQt),invQt))

    # hang on: haven't included the log prob of hyperparams, under hyperprior.
    (logProbTheta, gradLogProbTheta) = calcLogProbTheta(theta,hyperprior)

    return -(dLogL + gradLogProbTheta)  # minus because fmin_cg finds MINIMUM

def BROKEN_calcNegGradLogL(theta, args):
    """
    calc the negative gradient of the log likelihood of the data.
    """
    (n,D,K,Q,L,alpha) = doCholesky(theta,args)
    invQ = linalg.solve(transpose(L),  linalg.solve(L,eye(n)) )  
    # yep, I've checked and this seems to be inv(Q). Tick.

    # Precompute the matrix alpha*alpha^T - inv(Q).
    A = dot(alpha,transpose(alpha)) - invQ
    # NOT SURE I TRUST THIS.... not one of the grads is right, which
    # suggests the problem is in A. 
    dfX = zeros(shape(theta))
    for d in range(D):
        B = [X[:,d]] * ones((n,n))
        lengthScale = exp(theta[d])
        Sd = pow(B-transpose(B),2) * lengthScale
        grad = -Q * Sd
        dfX[d] = 0.5 * trace(dot(A,grad))
    grad = K
    dfX[D] = 0.5 * trace(dot(A,grad))
    noiseVar = exp(theta[D+1])
    grad = noiseVar * eye(n)
    dfX[D+1] = 0.5 * trace(dot(A,grad))
    # BUT THIS IS STILL BUG-RIDDEN AND BOGUS.
    return -dfX  

#-------------------------------------------------------------------------

def demo1Dplot(theta,args,outfile,colour=array([0,0,1.0])):
    faded = 1-(1-colour)/2.0

    (X,y,hyperprior) = args
    (n, D) = shape(X)

    xrange = X.max() - X.min()
    Xtest = arange(X.min()-xrange/2,X.max()+xrange/2,(X.max()-X.min())/100)
    Xtest.shape = (len(Xtest),1)

    mu,sig2 = calcGPPrediction(theta,args,Xtest)
    Xtest.shape = (len(Xtest),)
    noiseVar = exp(theta[D+1])  # it's the last hyperparameter in the list.
    sig2 = sig2 + noiseVar

    fig = pylab.figure()
    pylab.subplots_adjust(hspace=0.001)

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1,sharey=ax1)
    ax1.plot(X,y,'ok')
    ax1.plot(Xtest,mu,'-k')
    ax1.fill_between(Xtest,mu+sqrt(sig2),mu-sqrt(sig2),color=faded,alpha=.2)

    # plot the shape of the current MAP covariance function in use.
    #pylab.subplot(1,2,2)
    ### D==1, so exp(theta) is [lengthscale, verticalScale, noiseVar]
    covariance = exp(theta[1])*exp(-exp(theta[0])*pow(Xtest,2))
    ax2.fill_between(Xtest,0,covariance,color='black',alpha=.2)
    # Also show the noiseVar, to be added to the diagonal of Cov matrix
    ax2.plot([0], [exp(theta[1]) + exp(theta[-1])],'o',color='black')
    #pylab.ylabel('covariance')
    #pylab.xlabel('distance')

    pylab.setp(ax1.get_xticklabels() , visible=False)
    pylab.setp(ax2.get_yticklabels() , visible=False)
    pylab.draw()
    pylab.show()
    pylab.savefig(outfile)
    print 'Wrote %s.png ' % (outfile)


#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = str(sys.argv[-1])
    else:
        sys.exit('usage: python gp.py datafile')
    data = loadtxt(filename)
    print data
    X = data[:,0:-1] # everything except the last column
    y = data[:,-1]   # just the last column

    # rng.seed(1101) # comment this out to get a random seed.


    (n, D) = shape(X)
    (theta, hyperprior) = setAndSampleHyperprior(D)

    args = (X,y,hyperprior) # fmin_cg needs these all in one box...
	

    print 'initial log likelihood: ',-calcNegLogPosterior(theta,args)
    print 'initial exp(theta): ', exp(theta)


    # check we haven't busted anything lately...!
    testGradientCalc = check_grad(calcNegLogPosterior, calcNegGradLogPosterior, theta, args)
    if testGradientCalc > 0.05:
        sys.exit("calcNegGradLogPosterior does not give correct grad of calcNegLogPosterior!")


    # update to better hyperparameters, using conjugate gradients algorithm.
    newTheta = fmin_cg(calcNegLogPosterior, theta, calcNegGradLogPosterior, [args], gtol=1e-4,disp=0)


    print 'final log likelihood: ',-calcNegLogPosterior(newTheta,args)
    print 'final exp(theta): ', exp(newTheta)

    if (D == 1):
        demo1Dplot(newTheta,args,'gp_test',array([0.7,0,0]))
