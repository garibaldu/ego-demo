import sys, pylab, gp
import numpy as np
from numpy.random import multivariate_normal as multivariate_normal

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    exptheta = np.array([1.0,1.0,0.01])
    if len(sys.argv) == 5:
        exptheta[1] = float(sys.argv[1])
        exptheta[0] = 1.0/pow(float(sys.argv[2]),2)
        exptheta[2] = pow(float(sys.argv[3]),2.0)
        outfile = sys.argv[-1]
    else:
        print 'usage: python makeSamples.py #1 #2  #3  outfilename'
        print '#1 is the vertical scale of Cov function'
        print '#2 is x-length scale'
        print '#3 is std dev of target noise'
        print 'eg:    python makeSamples.py 1 .1 .01  test'
        sys.exit('I quit')

    theta = np.log(exptheta)
    num_curves = 1
    XLim = 10.0
    num_points = 5000
    X = np.arange(0,XLim+.000001,XLim/num_points)
    X.shape = (len(X),1)

    """
    # make several sample curves from the prior
    K = gp.calcCovariance(X, X, theta)
    mean = np.zeros((len(X),),float)  
    z = multivariate_normal(mean, K, num_curves) 
    pylab.subplot(211)
    pylab.plot(X,np.transpose(z))
    pylab.title('Samples from a vanilla Gaussian process')
    """




    # now let's try out a sum of two covariance functions for fun...
    hypeB = np.array([1,3,0.001],float)  # i.e. big and wide.
    """ The next few lines look confusing, but it's just that the command line args come in as 
    vertical scale, horizontal scale and typical noise level, whereas they need to be converted
    to the form that everything else uses, which is 1st the INVERSE length scale, SQUARED 
    [and note there's only one of these as this demo is just for 1-D] 
    then the vertical scale (as is), and finally the VARIANCE of the noise level (ie. squared).
    AND on top of that, we store them in log space in order to be able to apply simple Gaussian 
    hyperpriors!
    """
    exptheta[1] = float(hypeB[0])
    exptheta[0] = 1.0/pow(float(hypeB[1]),2)
    exptheta[2] = pow(float(hypeB[2]),2.0)
    theta_B = np.log(exptheta)


    # Now we can make several sample curves from the prior
    K2 = gp.calcCovariance_superposition(X, X, theta, theta_B)
    mean = np.zeros((len(X),),float)  
    y = multivariate_normal(mean, K2, num_curves)
    y = np.ravel(y - np.min(y))
    print y.shape
    for i in range(num_points/5):
        y[i] = y[i]*i/(num_points/5.0)
        j = num_points-1 -i
        y[j] = y[j] * i/(num_points/5.0)

    #pylab.subplot(212)
    pylab.plot(X,np.transpose(y))
    pylab.title('A superposition of 2 Gaussian processes')
    pylab.savefig(outfile+'.png')
    print 'wrote %s.png' % (outfile)

    data = np.zeros((len(np.ravel(X)),2),float)
    data[:,0] = np.ravel(X)
    data[:,1] = np.transpose(y)
    np.savetxt(outfile+'.txt',data,fmt='%.3f')
    print 'wrote %s.txt' % (outfile)
