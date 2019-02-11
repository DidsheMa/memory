import sys
from StimulusPresentation import StimulusPresentation, PresentationMode, PresentationStatus
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, QEventLoop, QDateTime
from lslrecorder import Lslrecorder

class P3speller():
    def __init__(self):
        self.lslrec = Lslrecorder()
        if(self.lslrec.findStream() == -1):
            return
        self.lslrec.startRec()
        app = QApplication(sys.argv)
        presentation = StimulusPresentation()

        # init speller-mode with image path and inter-stimulus-interval,
        presentation.initialize('speller_images/', 120, PresentationMode.SPELLER_MODE)
        # show window
        presentation.show()
        # simulate waiting time to demonstrate startup-feature
        #curr_time = QDateTime.currentMSecsSinceEpoch()
        #while QDateTime.currentMSecsSinceEpoch() < curr_time + 1000:
        #    QCoreApplication.processEvents(QEventLoop.AllEvents, 100)
        # show target with text
        #presentation.show_target(1, 'Search for this Target')
        # execute subtrial and save flash-sequence
        _flash_sequence = presentation.execute_sub_trial()
        curr_time = QDateTime.currentMSecsSinceEpoch()
        while QDateTime.currentMSecsSinceEpoch() < curr_time + 1000:
            QCoreApplication.processEvents(QEventLoop.AllEvents, 100)

        # show target, and afterwards highlight target in matrix before next keypress
        #presentation.show_target(5, 'This is multi-\nline text', show_matrix=True)
        #_flash_sequence = presentation.execute_sub_trial()

        print(_flash_sequence)
        self.getDataSnippets(_flash_sequence)
        sys.exit(app.exec_())

    def getDataSnippets(self, flash_sequence):
        for (ind,timestamp) in flash_sequence:
            self.lslrec.getDataAtTimestamp(timestamp)
            return


if __name__ == '__main__':
    P3speller()
