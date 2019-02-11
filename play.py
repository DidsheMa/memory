import sys
from Memory import StimulusPresentation, PresentationMode, PresentationStatus
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, QEventLoop, QDateTime
from lslrecorder import Lslrecorder
import numpy as np
import time
from auxiliary import *
from termcolor import colored
import mne.filter as filter

class P3_classify():
    def __init__(self):
        self.prefix = "data/"
        self.images_prefix = 'speller_images/'
        self.samples = None
        self.hasPCA = False
        self.isi = 120 # speller stim interval in ms
        self.stimnum = 8
        self.trials = 20
        self.do_channel_mask = False
        self.mode = None

        self.loadData()

        if self.mode == PresentationMode.SPELLER_MODE:
            self.images = int((self.stimnum / 2) ** 2)
        else:
            self.images = self.stimnum

        self.lslrec = Lslrecorder()
        if(self.lslrec.findStream() == -1):
            return
        if self.do_channel_mask is True:
            self.channels = 9
        else:
            self.channels = self.lslrec.channelCount
        self.lslrec.startRec()
        app = QApplication(sys.argv)
        self.presentation = StimulusPresentation(quit_with_escape=True)

        # init speller-mode with image path and inter-stimulus-interval,
        self.presentation.initialize(path=self.images_prefix, inter_stimulus_interval=120,
                                     n_images=self.images, mode=self.mode)

        # show window
        self.presentation.show()

        # detect targets
        self.detectTargets()
        sys.exit(app.exec_())

    def detectTargets(self):
        other = -1 
        for i in range(self.trials):
            s = 1
            target = -1
            if self.mode == PresentationMode.SPELLER_MODE:
                TP = np.zeros((int(self.stimnum/2),int(self.stimnum/2)))
            else:
                TP = np.zeros((self.stimnum))
                print(colored("TP has length: %s " % len(TP), 'green'))

            #Show only blank matrix to allow user to choose symbol
            self.presentation.show_target(-2, '', show_matrix=True)

                # % Subtrial loop. Executed until a target is detected or max. number
                # % of subtrials is reached.
            while target == -1 :

                flash =self.presentation.execute_sub_trial()
                on = np.array([int(i[1]) for i in flash])
                fseq = np.array([int(i[0]) for i in flash])

                # wait for all data to be collected
                time.sleep(1)

                (target,TP) = self.processSubtrial(on ,fseq, s, TP)
                print(target)
                print(TP)
                s=s+1

            if self.mode == PresentationMode.SPELLER_MODE:
                #    % Expand row/col to linear index [0...(stimnum/2)^2-1]
                target = (target[1])*(self.stimnum/2)+target[0]
            print(colored("The current target has index %s." % target, 'magenta'))
            (index, turned) = self.presentation.fields[target]
            if not turned:
                 self.presentation.fields[target] = (index, True)
                 self.presentation.show_target(target, 'Detected Target')
                 if other >= 0:
                     if other != target:
                         self.presentation.fields[target] = (index, False)
                         (other_index, turned) = self.presentation.fields[other]
                         self.presentation.fields[other] = (other_index, False)
                 else:
                     other = target


    def processSubtrial(self, on, fseq, s, TP):
        self.dataset = np.zeros((self.stimnum,self.samples,self.channels))
        for (stim, onset) in zip(list(range(self.stimnum)),on):
            ind = (np.abs(self.lslrec.timeStamps - onset)).argmin()
            # print("closest timestamp: ",int(self.lslrec.timeStamps[ind]))
            thresh = (1000/self.lslrec.srate)*2
            diff = abs(int(self.lslrec.timeStamps[ind]) - onset)
            #print(diff, thresh)
            if diff > thresh:
                print('difference between timestamps is %d ms. Something is wrong' % diff)
                return
            self.dataset[stim] = self.getDataAtIndex(ind)
        #print(self.dataset.shape)

        if (not self.hasPCA):
            self.dataset = downsample(self.dataset, 10, avg="pick")
            #print("downsampled dataset:")
            #print(self.dataset.shape)

        self.dataset = self.dataset.transpose((0, 2, 1))
        self.dataset = self.dataset.reshape(self.stimnum, -1)

        # apply pca
        if self.hasPCA:
            self.dataset = np.matmul(self.dataset,self.pcamat)

        # apply classification
        ys = np.matmul(self.dataset,self.fda_w) + self.fda_b
        return self.subtrialDynamic(ys, fseq, TP, s)

    def subtrialDynamic(self, ys, flashseq, TPprev, subtrcount, maxRep = 10):

        # % function [target,TPcurr] = class_subtrial_dynamic(ys,flashseq,TPprev,sbtrcount,Mthresh,isSpeller,maxrep)
        # %
        # % THIS IS EXECUTED ONCE PER SUBTRIAL !
        # %
        # % INPUT
        # %   flashseq    [stimnum x 1]
        # %   ys          [stimnum x 1]
        # %   TPprev      Current score matrix
        # %   sbtrcount   Current repetion
        # %   Mthresh     Matrix brightness threshold
        # %   maxrep      Maximal number of subtrials
        # %
        # % OUTPUT
        # %   target      index of decoded stimulus

        dummy = np.array(list(zip(flashseq, ys))).transpose()
        dummy = dummy[:, np.argsort(dummy[0])]

        if self.mode == PresentationMode.SPELLER_MODE:
            rc = int(len(flashseq)/2)
            TPcurr = TPprev + np.tile(dummy[1][0:rc],(rc,1))+np.tile(dummy[1][rc:],(rc,1)).transpose()

            M = sum(Scale(TPcurr.flatten(),0,1))
            #print("sum over scaled TP")
            #print(M)

            max_idx = np.argmax(TPcurr)
            max_col = int(np.floor(max_idx / rc))
            max_row = int(max_idx-(max_col*rc))

            target = [max_col,max_row]
            #print(TPcurr[max_col,max_row])



        # TODO NOT speller case
        else:
            TPcurr = TPprev + dummy[1, :]
            M = sum(Scale(TPcurr.flatten(), 0, 1))
            print(colored("M = %s " % M, 'magenta'))
            print(colored("M_thr is %s " % self.M_thr, 'red'))
            print(colored("TPcurr has shape: %s: " % len(TPcurr), 'magenta'))
            print(TPcurr)
            target = np.argmax(TPcurr)
        # else

        #Check if we haven't met criteria yet and need to issue another subtrial
        #print("TPcurr")
        #print(TPcurr)
        if(M > self.M_thr and subtrcount<maxRep):
            target = -1
        print("M is currently %s" % M)
        return target, TPcurr

    def getDataAtIndex(self,i):
        returnData = self.lslrec.data[i:i+self.samples][:]
        if self.do_channel_mask is True:
            mask = [2, 3, 4, 5, 6, 7, 14, 15, 16]
            returnData = filter.filter_data(returnData[:, mask].T, self.srate, l_freq=0.1, h_freq=None, method='iir', n_jobs=6)
            returnData = returnData.T
        return returnData

    def loadData(self):
        if self.hasPCA:
            self.pcamat = np.load(self.prefix + "pcamat.npy")
        self.fda_w = np.load(self.prefix + "fda_w.npy")
        self.fda_b = float(np.load(self.prefix + "fda_b.npy"))
        self.Tp_thr = float(np.load(self.prefix + "Tp_thr.npy"))
        self.M_thr = float(np.load(self.prefix + "M_thr.npy"))
        print("Threshold for dyn lim is %s" % self.M_thr)
        self.srate = int(np.load(self.prefix + "srate.npy"))
        self.samples = int(0.8 * self.srate)
        self.mode = np.load(self.prefix + "presentation_mode.npy")
        print("loaded data from " + self.prefix)

if __name__ == '__main__':
    P3_classify()
