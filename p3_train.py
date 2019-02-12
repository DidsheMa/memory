import numpy as np
from auxiliary import *
import mne.filter as filter
from Memory import PresentationMode
from termcolor import colored

#################################################################################
# This script trains a classifier for the P300 Speller BCI.
# A directory data must contain the varianles:
# data.npy       flashseq.npy   onsets.npy     targets.npy    timestamps.npy
#################################################################################


class P3_train():
    def __init__(self):
        ##############################################8#############
        # USER CONFIGURATION:
        # SPECIFY PARAMETERS FOR THE EXPERIMENT
        ###########################################################
        self.prefix = "data/"
        self.do_channel_mask = False # needed when using SMARTING amplifier
        self.folding = 5  # should be multiple of trial number



        self.samples = None
        self.channels = None
        self.doPCA = False
        self.doERP = False
        self.mode = None


        # load recorded data
        self.loadData()
        #self.loadDataAf()
        # reconstruct key values
        (self.trials, self.subtrials, self.stimnum) = self.onsets.shape

        # print("data shape" + self.data.shape)
        if self.do_channel_mask is True:
            mask = [2,3,4,5,6,7,14,15,16]
            # print("channel mask" + mask)
            self.data = self.data[:, mask]
            self.data = filter.filter_data(self.data.T, self.srate, l_freq=0.1, h_freq=None, method='iir', n_jobs=6)
            self.data = self.data.T

        self.channels = self.data.shape[1]
        print("channel number %s" % self.channels)
        # print("data shaper after channel mask apply" + self.data.shape)

        self.onsets = self.onsets.flatten()
        # print(self.timestamps)
        self.epochs = self.onsets.shape[0]
        # print(self.onsets)

        # print(self.epochs)
        self.dataset = np.zeros((self.epochs,self.samples,self.channels))
        for (epoch, onset) in zip(list(range(self.epochs)),self.onsets):
            print("onset: ", (int(onset)))
            ind = (np.abs(self.timestamps - onset)).argmin()
            print("closest timestamp: ",int(self.timestamps[ind]))
            print("index: ",ind)
            self.dataset[epoch] = self.getDataAtIndex(ind)

        print("first timestamp ",int(self.timestamps[0]))
        print("last timestamp ", int(self.timestamps[len(self.timestamps)-1]))
        print("len timesstamps", len(self.timestamps))
        print("dataset ", self.dataset)


        print("dataset before ds",self.dataset.shape)
        if (not self.doPCA and not self.doERP):
            self.dataset = downsample(self.dataset, 10, avg="pick")
            print("downsampled dataset:")
        else:
            pass

        print("dataset after ds",self.dataset.shape)
        #print(self.dataset[0])

        self.dataset = self.dataset.transpose((0, 2, 1))

        if not self.doERP:
            print('before:', self.dataset.shape, 'after:', (self.epochs, self.samples * self.channels))

            #self.dataset = self.dataset.reshape(self.epochs, self.samples * self.channels)
            self.dataset = self.dataset.reshape(self.epochs, -1)
            self.samples = self.channels

        #print(self.dataset.shape)

        # extract labels, needs target and flashsequence
        self.label = np.zeros((self.trials, self.subtrials, self.stimnum))

        # print('ts', self.targets.shape)
        # print('trc', self.targets)
        if self.mode == PresentationMode.SPELLER_MODE:
            rc = self.stimnum / 2
            cols = np.mod(self.targets, rc)
            print(cols)
            rows = np.ceil(self.targets / rc)
            print(rows)
            self.targets = np.array([rows.transpose(), cols.transpose()])
            self.targets = self.targets.transpose()
            for i in range(0, self.trials):
                for j in range(0, self.subtrials):
                    for k in range(0, self.stimnum):
                        index = i*self.subtrials*self.stimnum + j*self.stimnum + k
                        #print('target %d: %s, %s, %s' % (index, str(self.flashseq[i][j][k]), cols[i], rows[i]))
                        self.label[i][j][k] = self.flashseq[i][j][k] == cols[i] + rc or self.flashseq[i][j][k] == rows[i]
                        #print('target %d: %s' % (index, str(self.label[i][j][k])))
        else:
            for i in range(0, self.trials):
                    for j in range(0, self.subtrials):
                        for k in range(0, self.stimnum):
                            if self.targets[i]==self.flashseq[i, j, k]:
                                self.label[i, j, k] = 1


        xvaltrials = int(self.trials/self.folding)

        # Stuff for holding TPscore related results
        print(str(xvaltrials))
        TPscore = np.zeros((self.folding, xvaltrials))
        M_tp = np.zeros((self.folding, xvaltrials))
        sbtrixs = np.zeros((self.folding, xvaltrials,2))

        countcorrect = 0
        ixs = list(range(self.folding))

        # % Indeces to extract train- and testset
        xvalixs = np.tile(range(self.folding), [xvaltrials,1]).transpose().reshape(self.folding*xvaltrials)
        xvalepochs = np.tile(range(self.folding), [xvaltrials*self.subtrials*self.stimnum,1]).transpose().reshape(self.folding*xvaltrials*self.subtrials*self.stimnum)

        self.label = self.label.reshape(len(xvalepochs))

        if self.doERP:
            import matplotlib.pyplot as plt
            posdata = self.dataset[self.label == 1]
            negdata = self.dataset[self.label != 1]
            plt.plot(np.mean(posdata[:, 1, :], axis=0))
            plt.plot(np.mean(negdata[:, 1, :], axis=0))
            plt.show()



        # %% XVALIDATION LOOP

        auc = np.zeros(self.folding,)
        for i in range(self.folding):
            print('\n ----- performing folding %d -----\n' % i)
            #     % Stuff for TRAINING
            trainset = self.dataset[xvalepochs!=i]
            trainlabel = self.label[xvalepochs!=i]
            #print('label', self.label)
            #print('trainlabel:', trainlabel)
            #print(trainset.shape)
            #print(trainset[0])

            if self.mode == PresentationMode.SPELLER_MODE:
                targetsfold = self.targets[xvalixs==i,:] #[xvaltrials x 2]
            else:
                targetsfold = self.targets[xvalixs==i]
            testset = self.dataset[xvalepochs == i, :]
            flashfold = self.flashseq[xvalixs==i,:,:] #[xvaltrials x subtrials x stimnum]
            testlabel = self.label[xvalepochs==i]

            #print("targetsfold" ,targetsfold)
            #print("flashfold", flashfold.shape)
            #print("testset", testset.shape)
            #print("testlabel", testlabel.shape)
            #print("trainset", trainset.shape)


            if self.doPCA:
                (pcamat,_,_) = pca(trainset.transpose())
                #print("pca mat shape", pcamat.shape)
                trainset = np.matmul(trainset, pcamat)
                testset = np.matmul(testset, pcamat)

            (fda_w, fda_b) = train_fda(trainset, trainlabel) # TRAIN
            #print("fda_w", fda_w)
            #print("fda_b", fda_b)
            ys = np.matmul(testset,fda_w)+fda_b # TEST

            #print('ys:', ys)
            for j in list(range(xvaltrials)):
                #print('from %d to %d' % (self.subtrials * self.stimnum * j , self.subtrials * self.stimnum * (j +1)))
                #print(ys[self.subtrials * self.stimnum * j : self.subtrials * self.stimnum * (j +1)])
                ys_trial = np.reshape(ys[self.subtrials * self.stimnum * j : self.subtrials * self.stimnum * (j +1)], (self.subtrials, self.stimnum))
                #print(ys_trial.shape)
                #np.save("ys_trial", ys_trial)
                #np.save("flashfold", flashfold)
                #np.save("targetsfold", targetsfold)

                print('tp',TPscore[i,j])
                print('mtp', M_tp[i,j])
                print('sbt', sbtrixs[i,j])

                if self.mode == PresentationMode.SPELLER_MODE:
                    print("Blaaaaaaaaaaaaaaaaaaaa")
                    (TPscore[i,j], M_tp[i,j], sbtrixs[i,j]) = class_calcTPscore(ys_trial, flashfold[j], targetsfold[j])
                else:
                    print("Grrrrrrrrrrrrrrrrrrr")
                    (TPscore[i, j], M_tp[i, j], sbtrixs[i, j]) = class_calcTPscore_sequence(ys_trial, flashfold[j],targetsfold[j])

            # #     % Calc AUC to check for accuracy
            # todo maybe fix auc funcs in auxiliary
            roc = calc_ROC(ys,100, testlabel)
            auc[i] = calc_AUC(roc)
            # print('roc', roc)
            print('auc', auc[i])

        print('mean auc', np.mean(auc))

        #FINAL CLASSIFIER TRAINING FOR USE IN TEST
        if self.doPCA:
            (pcamat,_,_) = pca(self.dataset.transpose())
            trainset = np.matmul(self.dataset, pcamat)
        else:
            trainset = self.dataset

        (fda_w, fda_b) = train_fda(trainset, self.label) # FINAL TRAIN

        # Calc mean TP and M to be used for dynamic subtrial limiting
        Tp_thr = np.mean(TPscore[TPscore != 0])
        M_thr = np.mean(M_tp[M_tp != 0])
        print(colored("Tp_thr is %s" % float(Tp_thr), 'magenta'))
        print(colored("M_thr is %s" % float(M_thr), 'magenta'))

        #save class data
        if self.doPCA:
            np.save(self.prefix + "pcamat", pcamat)
        np.save(self.prefix + "fda_w", fda_w)
        np.save(self.prefix + "fda_b", fda_b)
        np.save(self.prefix + "Tp_thr", Tp_thr)
        np.save(self.prefix + "M_thr", M_thr)
        print("saved data to " + self.prefix)


    def loadData(self):
        # load recorded data
        self.data = np.load(self.prefix + "data.npy")
        self.timestamps = np.load(self.prefix + "timestamps.npy")
        self.targets = np.load(self.prefix + "targets.npy")
        self.onsets = np.load(self.prefix + "onsets.npy")
        self.flashseq = np.load(self.prefix + "flashseq.npy")
        self.srate = int(np.load(self.prefix + "srate.npy"))
        self.channels = self.data.shape[1]
        self.samples = int(0.8 * self.srate)
        self.mode =  np.load(self.prefix + "presentation_mode.npy")
        print("loaded data from " + self.prefix)


    def loadDataAf(self):
        import scipy.io as sio
        import os
        datapath = '/vol/bci/p3bci/data/'

        # Load the mat file
        struct = sio.loadmat(os.path.join(datapath,'af-30trials_data.mat'))

        # Extract the variables contained in file
        self.data = struct['data']
        self.flashseq = struct['flashseq']
        self.onsets = struct['onsets']
        self.targets = struct['targets'][0]
        self.timestamps = struct['timestamps']
        self.isSpeller = struct['isSpeller']
        print(self.flashseq)
        print(self.targets)
        print('##############################')
        print('######LOADED##################')
        print('##############################')



    def getDataAtIndex(self,i):
        returnData = self.data[i:i+self.samples][:]
        print(returnData.shape)
        print("Current end index: %s." % (i+self.samples))
        print("Overall data length: %s." % self.data.shape[0])
        return returnData


if __name__ == '__main__':
    P3_train()
