from __future__ import division
import numpy as np
from math import floor
from numpy import matlib
from scipy import linalg

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

def class_calcTPscore(ys,flashseq,target,isSpeller=0,doImg=0):

    subtrials = ys.shape[0]
    M = np.zeros(ys.shape)

    if(isSpeller):
        rc = int(ys.shape[1]/2)
        print(type(subtrials), subtrials)
        print(type(rc), rc)
        M_sp = np.zeros((rc,rc,subtrials))

    for i in range(subtrials):
        dummy = np.array(zip(flashseq[i], ys[i])).transpose()
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
    else:
        TPscore  = 0
        M_tp = 0

    return TPscore, M_tp, sbtrix


def Scale(Data, Lower=-1, Upper=-1):
    if Lower > Upper:
       print(['Wrong Lower or Upper values!'])

    d=np.copy(Data)
    d-=d.min()
    d+=Lower
    d*=Upper/d.max()
    return d

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
