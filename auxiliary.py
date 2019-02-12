from __future__ import division
import numpy as np
from math import floor
from numpy import matlib
from scipy import linalg
from termcolor import colored

def downsample(data,fac,avg="pick",axis=0):
    """
    INPUT
        data    data array sized [trials x samples x channels] OR matrix
                 sized [trials x channels]
        fac     downsampling factor
        avg     string 'avg' (averaging, default) or 'pick' (no avg.)
    OUTPUT
        X       the downsampled signal array sized
                  [trials x floor(samples/fac) x channels]
    """

    tmp = np.zeros((2,4))
    dims = data.shape
    X = [];

    if len(dims)==3:
        samples = dims[1]
        nums = (int)(floor(samples/fac))
        if avg=="avg":
            X = np.zeros((dims[0],nums,dims[2]),np.float64)
            for i in range(nums):
                X[:,i,:] = np.mean(data[:,i*fac:fac+i*fac,:],axis=1)
        elif avg=="pick":
            X = data[:,range(fac-1,fac*nums,fac),:]

    elif len(dims)==2:
        samples = dims[0]
        nums = (int)(floor(samples/fac))
        if avg=="avg":
            X = np.zeros((nums,dims[1]),np.float64)
            for i in range(nums):
                X[i,:] = np.mean(data[i*fac:fac+i*fac,:],axis=0)

        elif avg=="pick":
            X = data[range(fac-1,fac*nums,fac),:]
    else:
        print("wrong structure of input array! Required data array sized"
        + "trials x samples x channels OR matrix sized trials x channels.")

    return X


# TODO REMOVE FOR UEBUNG
def train_fda(data,label,trainerr=False):
    """
    Train binary FDA classifier. Obtain params fda_w and fda_b.
    INPUT
        data        Training data. ndarray (trials,features)
        label       Class label {-1,1} or {0,1}. ndarray (trials,1)
        trainerr    Flag. Wether to return the training error (default False)
    OUTPUT
        fda_w       Weight vector. ndarray (features,1)
        fda_b       Bias (i.e., the class boundary). scalar
    """
    print("label size: ",label.shape)
    label = label.flatten()

    if data.shape[0]!=label.shape[0]:
        data = np.transpose(data)

    if 0 in label:
        ixs = label==0
        label[ixs] = -1

    #print("label: ",label[0])
    #print("data: ",data[0])

    posdata = data[label==1,:]
    negdata = data[label==-1,:]

    #print("label == 1", (label==1))

    posmean = np.mean(posdata,axis=0)
    negmean = np.mean(negdata,axis=0)

    #print("pos: ",posdata[0])
    #print("neg: ",negdata[0])

    #print("pos mean: ",posmean[0])
    #print("neg mean: ",negmean[0])

    Spos = cov_shrinkage(posdata)
    Sneg = cov_shrinkage(negdata)
    Sw = Spos + Sneg

    invSw = linalg.inv(Sw)
    fda_w = np.matmul(invSw,np.transpose(posmean-negmean))
    print("fda.w size: ",fda_w.shape)

    fda_posmean = np.matmul(np.transpose(fda_w),np.transpose(posmean))
    fda_negmean = np.matmul(np.transpose(fda_w),np.transpose(negmean))
    fda_b = (fda_negmean+fda_posmean)/2.0

    # ToDo: Block below not tested and fda_gamma not returned
    # if trainerr:
    #     testlabel = np.sign(np.matmul(data,fda_w)-fda_b)
    #     errixs = testlabel*label #-1 if error, 1 else
    #     errnum = np.nonzero(test<0)[0].shape[0]
    #     fda_gamma = 1-errnum/label.shape[0]
    #     return fda_w,fda_b,fda_gamma
    # else:
    #     return fda_w,fda_b

    return fda_w,fda_b

def cov_shrinkage(data,axis=0):
    """
    Implements algorithm for a shrinkage estimate of the covariance matrix.
    Uses the procedure as described in Schaefer and Strimmer (2005), which follows the
    Ledoit-Wolf Theorem.
    See P. 11, Target B  and Appendix
    INPUT
       data     ndarray (samples,dims) when axis=0 or (dims,samples) and axis=1
       axis     Wether features are contained in colums (default) or rows
    OUTPUT
       U       shrinkage estimate of covariance matrix (dims,dims)
    """
    print('covshape', data.shape)
    if axis==1:
        data = np.transpose(data)

    [rows,cols] = data.shape

    x_bar = np.sum(data,axis=0)/rows
    x_bar = np.matlib.repmat(x_bar,rows,1)


    w_ki = data-x_bar #factors for w_ijk

    s_ij = np.zeros((cols,cols),dtype=np.float64)
    var_s_ij = np.zeros((cols,cols),dtype=np.float64)

    #TODO MAKE column loops parallel
    for i in range(cols):
        for j in range(i,cols):
            w_kij = w_ki[:,i] * w_ki[:,j]
            #print(2, w_kij.shape)
            w_bar_ij = np.sum(w_kij,axis=0)/rows
            s_ij[i,j] = w_bar_ij * (rows/(rows-1))
            var_s_ij[i,j] = np.sum((w_kij - w_bar_ij)**2) * (rows/(rows-1)**3)


    mu = np.mean(np.diag(s_ij))

    T = np.eye(cols) * mu

    lambdaval = np.sum(np.reshape(np.triu(var_s_ij,0),(1,-1)))

    denom = np.sum(np.reshape(np.power(np.triu(s_ij,1),2),(1,-1))) + np.sum(np.power(np.diag(s_ij)-mu,2))

    lambdaval = lambdaval/denom

    U = lambdaval*T + (1-lambdaval)*np.cov(data,rowvar=False)

    return U


# TODO remove whole func
def pca(X,thr=0.999,alldims=False):
    """
    Compute the principal components of a given matrix.
    """
    print("Make sure X has shape [features x observations] !")

    Xcov = np.cov(X)
    eigval,eigvec = np.linalg.eig(Xcov)
    ixs = np.argsort(-eigval) # Sort in descending order, this is not guaranteed by numpy
    eigvec = eigvec[:,ixs]
    eigval = eigval[ixs]

    eigval = eigval.real
    eigvec = eigvec.real

    # TODO REMOVE THRESHOLD MECHANICS
    nlambda = np.sum(eigval) # Sum of all eigenvalues
    klambda = np.cumsum(eigval)/nlambda
    ixs = np.nonzero(klambda>=thr)
    pcaix = ixs[0][0] # Index when variance threshold reached
    # pcaix = pcaix[0]

    if not alldims:
        print("Reducing pca matrix to ",pcaix," dimensions (=99.9% variance)")
        eigvec = eigvec[:,0:pcaix+1]
        eigval = eigval[0:pcaix+1]

    return eigvec,eigval,pcaix



# Add a TP calculation for SEQUENCE mode as an own function to avoid confusion!
def class_calcTPscore_sequence(ys, flashseq, target):
    subtrials = ys.shape[0]
    print(colored("Num subtrials: %s " %subtrials, 'yellow'))
    print(colored("Num stimuli: %s " %ys.shape[1], 'yellow'))

    M = np.zeros(ys.shape)
    for i in range(0, subtrials):
        dummy = np.array(list(zip(flashseq[i], ys[i]))).transpose()
        dummy = dummy[:, np.argsort(dummy[0])]
        M[i] = dummy[1]
    
    TPprogressive = np.cumsum(M, axis=0)
    print(colored("Shape of TPprogressive: [%s, %s]" % (TPprogressive.shape[0], TPprogressive.shape[1]), 'cyan'))
    for val in TPprogressive[0, :]:
        print(colored("%s" % val, 'red'))
    # Find the subtrial (index), where the max score is the true target
    maxval = np.amax(TPprogressive, axis=1)
    maxix = np.argmax(TPprogressive, axis=1)
    print(colored('maxval is %s ' % maxval, 'cyan'))
    print(colored('maxix is %s ' % maxix, 'cyan'))
    print(colored("The target is %s " % target, 'green'))
    sbtrix = np.nonzero(maxix == target)  # gives index of successful subtrial
    sbtrix = sbtrix[0]
    if len(sbtrix) > 0:
        sbtrix = sbtrix[0]  # take only the first
        TPscore = maxval[sbtrix]  # return the score at successfull subtrial

        # Calculate the brightness of the matrix for the winnning subtrial
        M_tp = sum(Scale(TPprogressive[sbtrix,:], 0, 1))
    else:
        sbtrix = -1
        TPscore = 0
        M_tp = 0
    print(colored("Returning sbtrix as %s " % sbtrix, 'magenta'))
    print(colored("...and M_Thr as %s" % M_tp, 'magenta'))
    return TPscore, M_tp, sbtrix


# Leave this fundction ONLY with SPELLER mode code !!!
def class_calcTPscore(ys,flashseq,target):
    print("Diving into class_calcTPscore...")
    subtrials = ys.shape[0]
    M = np.zeros(ys.shape)
    rc = int(ys.shape[1]/2)
    M_sp = np.zeros((rc,rc,subtrials))

    for i in range(subtrials):
        #print(flashseq[i].shape)
        #print(flashseq[i])
        #print(ys[i].shape)
        #print(ys[i])
        #print(type(zip(flashseq[i], ys[i])))
        #print(zip(flashseq[i], ys[i]))
        dummy = np.array(list(zip(flashseq[i], ys[i]))).transpose()
        #print(dummy.shape)
        dummy = dummy[:, np.argsort(dummy[0])]
        M[i] = dummy[1]
        M_sp[:, :, i] = np.tile(M[i][0:rc],(rc,1))+np.tile(M[i][rc:],(rc,1)).transpose()

    print(['Target [', target[0], target[1], ']'])

    M_sum = np.zeros((rc, rc))
    sbtrix = -1
    for s in range(subtrials):
        M_sum = M_sum + M_sp[:,:,s]
        max_idx = np.argmax(M_sum)
        max_col = int(np.floor(max_idx / rc))
        max_row = int(max_idx-(max_col*rc))

        print('subtrial %d max:[%d,%d], target: %r' % (s, max_row, max_col, target))
        if max_col==target[1] and max_row==target[0]:
            print(['Found correct row and col in subtrial No. ',str(s)])
            sbtrix = s
            break
    if sbtrix != -1:
        TPscore = M_sum[max_col, max_row]
        M_tp = sum(Scale(M_sum.flatten(), 0, 1))
        print("Found correct target.")
        print("returning M_thr as %s" % M_tp)
        print("returning TPscore as %s" % TPscore)
    else:
        print("Did not find the correct target yet!")
        TPscore  = 0
        M_tp = 0
    return TPscore, M_tp, sbtrix


def Scale(Data, Lower=-1, Upper=-1):
    if Lower > Upper:
       print(['Wrong Lower or Upper values!'])

    d = np.copy(Data)
    d -= d.min()
    d += Lower
    d *= Upper/d.max()
    return d


# TODO REMOVE
def calc_ROC(classproj,steps, truelabel):
    """
    Calculate the Receiver Operator Characteristics (ROC) for given real-valued
    classification outputs (binary approach only).
    INPUT
        classproj   ndarray (vals,1)
        steps       int
    OUTPUT
    """

    # print('tl:', truelabel)
    thr = np.linspace(0,1,steps)
    roc = np.zeros((2,steps),dtype=np.float64)

    classproj = classproj.flatten()
    classproj = Scale(-1*classproj,0,1)  #0<=proj<=1

    # print('cp:', classproj)
    pred = np.zeros((classproj.shape[0],1),dtype=np.float64)

    for b in range(1,thr.shape[0]):
        ixs = (classproj<=thr[b]).nonzero()
        pred[ixs] = 1

        TP,FP,FN,TN = calc_confusion(pred,truelabel,1,0, mat=False)

        roc[0,b] = TP/len(truelabel[truelabel==0])
        roc[1,b] = FP/len(truelabel[truelabel!=0])

    return roc


def calc_confusion(testlabel,truelabel,pos=1,neg=0,mat=True):
    """
    Calc the confusion matrix. Currently only for binary class.
    """
    # Init confmat entries
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    testlabel = testlabel.flatten()
    truelabel = truelabel.flatten()
    posixs = np.nonzero(truelabel==pos)[0]
    negixs = np.nonzero(truelabel==neg)[0]

    for l in range(posixs.shape[0]):
        if testlabel[posixs[l]]==truelabel[posixs[l]]:
            TP+=1
        else:
            FN+=1
    for l in range(negixs.shape[0]):
        if testlabel[negixs[l]]==truelabel[negixs[l]]:
            TN+=1
        else:
            FP+=1

    if mat:
        confmat = np.array([[TP,FP],[FN,TN]])
        return confmat
    else:
        return TP,FP,FN,TN

def calc_AUC(roc):
    """
    Calc the Area under Curve (AUC) based on a/some given ROC(s).
    """
    if len(roc.shape)==3:
        auc = np.zeros((roc.shape[0],),dtype=np.float64)
        for f in range(roc.shape[0]):
            auc[f] = np.trapz(roc[f,0,:],roc[f,1,:])
    else:
        auc = np.trapz(roc[0,:],roc[1,:])

    return auc
