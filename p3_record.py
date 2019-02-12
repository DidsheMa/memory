import sys
from Memory import StimulusPresentation, PresentationMode, PresentationStatus
from PyQt5.QtWidgets import QApplication
from lslrecorder import Lslrecorder
from random import shuffle
import numpy as np
import time

#################################################################################
# This script runs a P300 Speller in "Training" Mode, records the data grabbed
# from an LSL stream, and saves data, timestamps, stimulus flash sequence,
# stimulus onsets, and targets to a directory data.
#################################################################################

class P3_train():
    def __init__(self):
        ###########################################################
        # USER CONFIGURATION:
        # SPECIFY PARAMETERS FOR THE EXPERIMENT
        ###########################################################
        self.data_prefix = 'data/' # Full path to directory where to save the data
        self.images_prefix = 'speller_images/' # Full path to directory where the stimulus images are located
        self.trials = 5
        self.subtrials = 10
        self.stimnum = 8
        # Possible modes:
        # PresentationMode.SPELLER_MODE (0): Stimuli arranged as square matrix. Rows and columns are highlighted.
        # PresentationMode.SEQUENCE_MODE (1): Stimuli are arranged side by side in middle of screen, each highlighted individually.
        # PresentationMode.CIRCULAR_SEQUENCE_MODE (2): Stimuli are arranged in a circle and highlighted individually.
        # Hint: Data processing for mode 1 and 2 is identical.
        self.mode = PresentationMode.SPELLER_MODE

        ###########################################################
        # DO NOT CHANGE CODE BELOW UNLESS YOU REALLY KNOW WHAT
        # YOU ARE DOING !!!
        ###########################################################

        self.images = int(((self.stimnum/2) ** 2))

        # important lists
        self.onsets = []
        self.flashseq = []
        self.targets = []
        while len(self.targets) < self.trials:
            self.targets += list(range(0, self.stimnum*2))
        shuffle(self.targets)
        self.targets = self.targets[0: self.trials]

        # create lsl recorder and start rec
        # ording data + timestamps
        self.lslrec = Lslrecorder()
        print("Created lsl recorder")
        if self.lslrec.findStream(hostname="dynamite") == -1:
            return
        self.lslrec.startRec()

        # start qt app and play experiment
        app = QApplication(sys.argv)
        self.presentation = StimulusPresentation()
        self.doPresentation()

        # shut everything down and save gathered data to data directory
        time.sleep(2)
        self.lslrec.stopRec()
        self.saveData()
        sys.exit(app.exec_())

    def doPresentation(self):
        # init speller-mode with image path and inter-stimulus-interval,
        # number of stimuli to load, and presentation mode
        self.presentation.initialize(path=self.images_prefix, inter_stimulus_interval=120,
                                     n_images=int(self.images/2), mode=self.mode)
        self.presentation.show()

        for trial in range(0, self.trials):
            print("displaying trial " + str(trial) + " " + str(self.targets[trial]))
            self.presentation.show_target(self.targets[trial], 'Focus on this Target', True)
            for subtrial in range(0, self.subtrials):
                flash = self.presentation.execute_sub_trial()
                # print("generated flash sequence with length: " + str(len(flash)))
                print("flash sequence content: " + str(flash))

                if trial == 0 and subtrial == 0:
                    self.flashseq = np.zeros((self.trials, self.subtrials, len(flash)))
                if subtrial == 0:
                    self.onsets.append([])

                self.flashseq[trial, subtrial, :] = [int(i[0]) for i in flash]
                self.onsets[trial].append([int(i[1]) for i in flash])
                print("flashseq" + str(self.flashseq.shape))



    def saveData(self):
        np.save(self.data_prefix + "data", self.lslrec.data)
        np.save(self.data_prefix + "timestamps", self.lslrec.timeStamps)
        np.save(self.data_prefix + "targets", self.targets)
        np.save(self.data_prefix + "onsets", self.onsets)
        np.save(self.data_prefix + "flashseq", self.flashseq)
        np.save(self.data_prefix + "srate", self.lslrec.srate)
        np.save(self.data_prefix + "presentation_mode", self.mode)
        print("saved data to " + self.data_prefix)


if __name__ == '__main__':
    P3_train()
