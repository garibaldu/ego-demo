"""
demo-gpo.py  - demonstration of optimization using Gaussian process regression. 
Marcus Frean. 

usage: python demo-gpo.py [datafile]

Reads a initial set of input vectors (x) from the file. Output (y)
values for these (and any future) x are given by one of the functions
in testFunctions.py

The task is then to choose where to sample next.
Each successive sample (x,y) taken in this way is added to the dataset.
"""

import testFunctions, gp, gpo, math, sys
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import fmin, fmin_cg, check_grad
from scipy.special import erf, erfc


class Index:
    ind = 0
    def sample(self, event):
        take_sample()
        update_figure()
    def update(self, event):
        update_model()
        update_figure()
    def sample_and_update(self, event):
        take_sample()
        update_model()
        update_figure()
    def back_one_sample(self, event):
        global X,Y
        X = X[:-1]
        Y = Y[:-1]
        update_model()
        update_figure()
    def toggle_ground_truth(self, event):
        global ground_truth_visible
        ground_truth_visible = not(ground_truth_visible)
        if ground_truth_visible:
            Line_ground_truth.set_alpha(1.0)
        else:
            Line_ground_truth.set_alpha(0.0)
        update_figure()

    def get_new_data(self, event):
        read_new_file()
        setup_plots()
        update_model()
        update_figure()

            
    
def update_figure():
    # Now update the plot with the new model's predictions
    Line_mean_pred.set_ydata(np.ravel(Ytest))
    errorbar_hi = np.ravel(Ytest+np.sqrt(Ysig2))
    errorbar_lo = np.ravel(Ytest-np.sqrt(Ysig2))
    Line_mean_plus_std.set_ydata(errorbar_hi)
    Line_mean_minus_std.set_ydata(errorbar_lo)
    Line_data_samples.set_xdata(np.ravel(X))
    Line_data_samples.set_ydata(np.ravel(Y))
    Line_EI.set_ydata(np.ravel(EI))

    Latest_point.set_xdata(X[-1])
    Latest_point.set_ydata(Y[-1])

    plt.draw()


def update_model(): # gets called when "update" button is pressed.
    global theta, best_so_far, EI,Ytest,Ysig2
    # set the arguments the optimizers need.
    thetaArgs = (X.reshape(len(X),1),Y,hyperprior) # fmin_cg needs these all in one box.
    # find MAP theta (i.e. hyperparameters), given all the available data.
    theta = fmin_cg(gp.calcNegLogPosterior,theta, gp.calcNegGradLogPosterior, [thetaArgs], gtol=1e-2,disp=0)

    # find the highest mean prediction, out of all the sample locations so far
    mus,sig2s = gp.calcGPPrediction(theta,thetaArgs,X.reshape(len(X),1))
    best_so_far = mus.max() # WAS best_so_far = Y.max()

    # Now we want to calc the mean, var, and EI at all the Xtest points.
    Ytest,Ysig2 = gp.calcGPPrediction(theta,thetaArgs,Xtest)
    EI = calcEI(Ytest,Ysig2,best_so_far)


def take_sample():
    global X,Y,yoffset

    # identify the point with highest EI
    i = np.argmax(EI)
    nextx = (Xtest[i])[0]
    print 'next x is %.1f (index %d, with EI %f = %f)' % (nextx,i,EI[i], np.max(EI))

    if nextx in X:
        print 'There has already been a sample there...'
        return #ie. we don't want to allow 2 samples at same place: ill-conditioning!
    # Store x (i.e. increase X by one row), and store y in Y.
    l = list(X)
    l.append(nextx)
    n=len(l)
    X = np.asarray(l)
    l = list(Y+yoffset) # the +yoffset reconverts back to true Y
    y = grnd_truth[i]
    l.append(y)
    Y = np.asarray(l)
    
    # Hack alert: moving Y up/down semi-arbitrarily!!!
    #yoffset = Y.mean()
    #Y = Y-yoffset


def calcEI(Ytest,Ysig2,best_so_far):
    Ysig = np.sqrt(Ysig2)
    u = (Ytest - best_so_far)/Ysig
    Phi = 0.5*erf(u/math.sqrt(2)) + 0.5
    phi = 1.0/math.sqrt(2*math.pi) * np.exp(-pow(u,2)/2.0)
    ei = Ysig*(u*Phi  + phi)

    # let's try smoothing it a bit...
    smoothed = np.zeros(ei.shape,float)
    for i in range(len(ei)):
        a = max(0, i-5)
        b = min(len(ei), i+5)
        smoothed[i] = np.mean(ei[a:b])
    return smoothed


def read_new_file():
    global Xtest, Ytest, Ysig2, grnd_truth, X, Y, thetaArgs, theta, EI, yoffset, hyperprior

    filename = filestem + str(rng.randint(10)) + '.txt'
    #filename = filestem + '.txt'
    print 'This is %s' % (filename)
    data = np.loadtxt(filename)
    Xtest = data[:,0:-1] # everything except the last column
    grnd_truth = data[:,-1]   # just the last column

    D = 1 # dimensionality of search space
    yoffset = 0.0 # to keep track of shifts up and down to y-axis
    initNumSamples = 2 # number of initial inputs
    #xlo,xhi =0.01,4.0 # just the plotting boundary, not a real constrait :(  
    #Xtest = np.arange(xlo,xhi,(xhi-xlo)/100.0) # the inputs we'll keep track of.
    d = len(np.ravel(Xtest))
    Xtest = Xtest.reshape((d,1))


    # take the initial samples
    X = np.zeros((initNumSamples)) 
    Y = np.zeros((initNumSamples,))
    for s in range(initNumSamples):
        i = rng.randint(0,d)
        X[s] = Xtest[i]
        Y[s] = grnd_truth[i]
    initX,initY = X,Y

    # Here we initialise hyperparameters, and make initial predictions.
    (init_theta, hyperprior) = gp.setAndSampleHyperprior(D) # initial hyperparams
    thetaArgs = (X.reshape(len(X),1),Y,hyperprior) # fmin_cg needs these all in one box.
    theta = fmin_cg(gp.calcNegLogPosterior,init_theta, gp.calcNegGradLogPosterior, [thetaArgs], gtol=1e-2,disp=0)
    Ytest,Ysig2 = gp.calcGPPrediction(theta,thetaArgs,Xtest)
    EI = calcEI(Ytest,Ysig2,np.max(Y))




def setup_plots():
    # Here we set up the plots
    global Line_EI, Line_mean_pred, Line_mean_plus_std, Line_mean_minus_std
    global Latest_point, Line_data_samples, Line_ground_truth
    global callback, button_update, button_sample, button_sampleupdate, button_back, button_ground, button_newfile
    
    plt.clf()
    #ax2 = plt.subplot(212)
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.6])
    plt.subplots_adjust(bottom=0.3)
    Line_mean_pred, = plt.plot(Xtest, Ytest, lw=10, alpha=.08)
    gp_colour = np.array([.025,.025,.9])
    errorbar = np.ravel(np.sqrt(Ysig2))
    Line_mean_plus_std, = plt.plot(Xtest, Ytest+errorbar,':',color=gp_colour,alpha=.5, lw=5)
    Line_mean_minus_std, = plt.plot(Xtest,Ytest-errorbar,':',color=gp_colour,alpha=.5, lw=5)
    Latest_point, = plt.plot(X[-1],Y[-1],'or',markersize=28)
    Line_data_samples, = plt.plot(X, Y, 'ok',markersize=20)
    Line_ground_truth = None
    Line_ground_truth, = plt.plot(Xtest,grnd_truth,'-k',alpha=0.0)
    global ground_truth_visible
    ground_truth_visible = False
    plt.ylabel('Value',fontsize=20)
    
    ax1 = plt.axes([0.05, 0.7, 0.9, 0.25])
    #ax1 = plt.subplot(211)
    Line_EI, = plt.plot(Xtest, EI,'g-', lw=3)
    plt.axis([np.min(Xtest),np.max(Xtest),0.0,1.5*np.max(EI)])
    plt.axis('off')
    plt.ylabel('Expected improvement')
    

    # Now we set up buttons so we can update stuff by clicking on the canvas.
    callback = Index()
    ybot = .01
    ax_update = plt.axes([0.1, ybot, 0.1, 0.075])
    button_update = Button(ax_update, 'Update')
    button_update.on_clicked(callback.update)
    ax_sample = plt.axes([0.25, ybot, 0.1, 0.075])
    button_sample = Button(ax_sample, 'Sample')
    button_sample.on_clicked(callback.sample)
    ax_sampleupdate = plt.axes([0.4, ybot, 0.1, 0.075])
    button_sampleupdate = Button(ax_sampleupdate, 'Next')
    button_sampleupdate.on_clicked(callback.sample_and_update)
    ax_back = plt.axes([0.55, ybot, 0.1, 0.075])
    button_back = Button(ax_back, 'Back')
    button_back.on_clicked(callback.back_one_sample)

    ax_ground = plt.axes([0.7, ybot, 0.1, 0.075])
    button_ground = Button(ax_ground, 'Truth')
    button_ground.on_clicked(callback.toggle_ground_truth)
    
    ax_newfile = plt.axes([0.85, ybot, 0.1, 0.075])
    button_newfile = Button(ax_newfile, 'New')
    button_newfile.on_clicked(callback.get_new_data)
    
    plt.show()


#--------------------------------------------------------------
# Main program starts here
#--------------------------------------------------------------

if __name__ == '__main__':
    global Xtest, Ytest, Ysig2, grnd_truth, X, Y, thetaArgs, theta, EI, yoffset, hyperprior

    global Line_EI, Line_mean_pred, Line_mean_plus_std, Line_mean_minus_std
    global Latest_point, Line_data_samples, Line_ground_truth
    global callback, button_update, button_sample, button_sampleupdate, button_back, button_ground, button_newfile

    if len(sys.argv) <= 1:
        sys.exit('usage: python demo-gpo.py datafile_STEM (e.g. \'test\')')

    filestem = str(sys.argv[-1])
    rng.seed(1) # comment this out to get a random seed.
    ground_truth_visible = False

    read_new_file()
    setup_plots()
