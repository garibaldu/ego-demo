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


    # make several sample curves from the prior
    K = gp.calcCovariance(X, X, theta)
    mean = np.zeros((len(X),),float)  
    z = multivariate_normal(mean, K, num_curves) 
    pylab.subplot(211)
    pylab.plot(X,np.transpose(z))
    pylab.title('Samples from a vanilla Gaussian process')
    pylab.savefig(outfile+'.png')
    print 'wrote %s.png' % (outfile)

    data = np.zeros((len(np.ravel(X)),2),float)
    data[:,0] = np.ravel(X)
    data[:,1] = np.transpose(y)
    np.savetxt(outfile+'.txt',data,fmt='%.3f')
    print 'wrote %s.txt' % (outfile)
