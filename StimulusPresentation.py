import math
import numpy as np
from enum import Enum
from random import shuffle
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QFontMetrics, QImage, QPainter
from PyQt5.QtCore import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from termcolor import colored
from pylsl import local_clock


class PresentationMode(Enum):
    SEQUENCE_MODE = 0
    SPELLER_MODE = 1
    CIRCULAR_SEQUENCE_MODE = 2


class PresentationStatus(Enum):
    STARTING_UP = -1
    IDLE = 0
    TARGET_SHOW = 1
    SUB_TRIAL_IN_PROGRESS = 2


class StimulusPresentation(QOpenGLWidget):

    def __init__(self, fullscreen=True, quit_with_escape=False):

        # setMouseTracking(true)
        print("stimpres initialisation")
        self.STATUS = PresentationStatus.STARTING_UP
        self.HIGHLIGHT_COL = -1
        self.MODE = PresentationMode.SPELLER_MODE
        self.TARGET_INDEX = -1
        self.MESSAGE = ''
        self.image_path = 'speller_images/'
        self.images = dict()
        self.image_values = dict()
        self.imageOrder = []
        self.files = []
        self.rows = 1
        self.cols = 1
        self.interStimulusInterval = 120
        self._key_pressed = False
        self.flash_pause = 50  # FIXED
        super(StimulusPresentation, self).__init__()
        self.quit_with_escape = quit_with_escape
        if fullscreen:
            self.setWindowState(Qt.WindowFullScreen)
        print("stimpres initialised")

    def initializeGL(self):
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        # glEnable(GL_POLYGON_SMOOTH);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 0)
        self.load_images()

    def resizeGL(self, w, h):
        width = w
        height = h
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #  glOrtho(0, w, 0, h,-1,1) # set origin to bottom left corner
        #  glOrtho(w, 0, 0, h,-1,1) # set origin to bottom right corner
        glOrtho(0, w, h, 0,-1,1) # set origin to top right corner
        #  glOrtho(w, 0, h, 0,-1,1) # set origin to top left corner
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initialize(self, path, inter_stimulus_interval, mode, n_images=None):
        if not path.endswith('/'):
            path += '/'
        print("initialize")
        self.MODE = mode
        self.image_path = path
        directory = QDir(self.image_path)
        name_filter = ['*.png',
                       '*.jpg',
                       '*.jpeg',
                       '*.bmp']

        if directory.exists():
            self.files = directory.entryList(name_filter)
            print("folder " + self.image_path + " does exist - found " + str(len(self.files)) + " files")
            if len(self.files) < 1:
                print("folder " + self.image_path + " is empty")
                return -1
            if n_images is not None:
                if len(self.files) < n_images:
                    print("folder " + self.image_path + " contains only " + str(
                        len(self.files)) + ",but experiment was specified to have " + str(n_images) + " stimuli")
                    return -1
                if len(self.files) > n_images:
                    print("folder " + self.image_path + " contains " + str(
                        len(self.files)) + ",but experiment is using only " + str(n_images) + " stimuli")
                    self.files = self.files[0:n_images]
        else:
            print("folder " + self.image_path + " does not exist")
            return -1
        self.interStimulusInterval = inter_stimulus_interval
        print("initialize finished")

    def show_target(self, stimulus_index, message='', show_matrix=False, skip_wait=False):

        if stimulus_index >= -1:
            self._key_pressed = False
            self.STATUS = PresentationStatus.TARGET_SHOW
            self.TARGET_INDEX = stimulus_index
            self.MESSAGE = message
            self.repaint()
            while not self._key_pressed and not skip_wait:
                QCoreApplication.processEvents(QEventLoop.AllEvents, 100)
        else:
            self.TARGET_INDEX = -1

        self._key_pressed = False

        # PAINT OBJECT MATRIX AND HIGHLIGHT TARGET
        if show_matrix:
            self.STATUS = PresentationStatus.SUB_TRIAL_IN_PROGRESS
            mode = self.MODE
            if mode == PresentationMode.SEQUENCE_MODE or mode == PresentationMode.SPELLER_MODE:
                self.MODE = PresentationMode.SEQUENCE_MODE
            elif mode == PresentationMode.CIRCULAR_SEQUENCE_MODE:
                self.MODE = PresentationMode.CIRCULAR_SEQUENCE_MODE
            self.repaint()
            self.MODE = mode
            while not self._key_pressed and not skip_wait:
                QCoreApplication.processEvents(QEventLoop.AllEvents, 100)
            self._key_pressed = False
            self.TARGET_INDEX = -1
            self.repaint()

        self.STATUS = PresentationStatus.IDLE

    def execute_sub_trial(self):
        flash_sequence = []
        self.STATUS = PresentationStatus.SUB_TRIAL_IN_PROGRESS
        self.TARGET_INDEX = -1

        # SUB-TRIAL LOOP
        order = []

        if self.MODE == PresentationMode.SPELLER_MODE:
            max_idx = self.cols + self.rows

        elif self.MODE == PresentationMode.SEQUENCE_MODE or self.MODE == PresentationMode.CIRCULAR_SEQUENCE_MODE:
            max_idx = len(self.files)
        else:
            # ERROR!!
            return

        for i in range(0, max_idx):
            order.append(i)

        shuffle(order)

        current_time = QDateTime.currentMSecsSinceEpoch()
        for i in order:
            if self.MODE == PresentationMode.SPELLER_MODE:
                self.HIGHLIGHT_COL = i >= self.rows
                self.TARGET_INDEX = i-self.rows if self.HIGHLIGHT_COL else i

            elif self.MODE == PresentationMode.SEQUENCE_MODE or self.MODE == PresentationMode.CIRCULAR_SEQUENCE_MODE:
                self.TARGET_INDEX = i

            else:
                pass
                # ERROR!!
            current_time_last = current_time
            current_time = QDateTime.currentMSecsSinceEpoch()

            #print("current_time diff to last current_time " + str(current_time - current_time_last))
            # print('stimpres', current_time)
            self.repaint()

            # use lsl localclock to be syncron with data
            # convert to ms and round to int
            flash_sequence.append((i, int(local_clock() * 1000)))

            while QDateTime.currentMSecsSinceEpoch() < current_time + (self.interStimulusInterval-self.flash_pause):
                QCoreApplication.processEvents(QEventLoop.AllEvents, 100)

            #print("time to flash " + str(QDateTime.currentMSecsSinceEpoch() - current_time))

            self.TARGET_INDEX = -1
            self.repaint()
            while QDateTime.currentMSecsSinceEpoch() < current_time + self.interStimulusInterval:
                QCoreApplication.processEvents(QEventLoop.AllEvents, 100)

            #print("time to flash + flash pause " + str(QDateTime.currentMSecsSinceEpoch() - current_time))

        self.STATUS = PresentationStatus.IDLE
        return flash_sequence

    def render_text(self, x, y, text, font=QFont('Arial', 50), color=Qt.white,
                    h_align=Qt.AlignLeft, v_align=Qt.AlignBaseline):
        fm = QFontMetrics(font)
        fr = fm.boundingRect(text)

        if h_align == Qt.AlignRight:
            x -= fr.width()
        elif h_align == Qt.AlignHCenter or h_align == Qt.AlignCenter:
            x -= fr.width()/2
        elif h_align == Qt.AlignLeft:
            pass
        else:
            print("WARNING: %r is not a valid option for horizontal text alignment. Set to Qt.AlignLeft." % h_align)

        if v_align == Qt.AlignTop:
            y += fm.ascent()
        elif v_align == Qt.AlignBottom:
            y -= fm.descent()
        elif v_align == Qt.AlignCenter or v_align == Qt.AlignVCenter:
            y += ((fr.height() / 2) - fm.descent())
        elif v_align == Qt.AlignBaseline:
            pass
        else:
            print("WARNING: %r is not a valid option for vertical text alignment. Set to Qt.AlignBaseline." % v_align)

        painter = QPainter(self)
        painter.setPen(color)
        painter.setFont(font)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.drawText(x, y, text)  # z = pointT4.z + distOverOp / 4
        painter.end()

    def paintGL(self):
        if self.STATUS == PresentationStatus.STARTING_UP:
            print("skipping paintgl")
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        if self.STATUS == PresentationStatus.TARGET_SHOW:  # SHOW TARGET
            msgs = self.MESSAGE.split('\n')
            font = QFont('Arial', 50)
            font.setBold(True)
            fm = QFontMetrics(font)

            if self.TARGET_INDEX >= 0:
                print(colored("Target index is %s." % self.TARGET_INDEX, 'cyan'))
                im = self.images[self.imageOrder[int(self.TARGET_INDEX)]]

                side = min(self.width(), self.height()*0.5)

                preserve_aspect = True
                if preserve_aspect:
                    im_path = ''
                    for k in self.images.keys():
                        if self.images[k] == im:
                            im_path = k
                            break
                    im_dim = self.image_values[im_path]
                    if im_dim.width() > im_dim.height():
                        w = side
                        h = int(float(side*(float(im_dim.height())/float(im_dim.width()))))
                    else:
                        h = side
                        w = int(float(side*(float(im_dim.width())/float(im_dim.height()))))
                else:
                    w = side
                    h = w

                self.display_image(im, (self.width() - w) / 2, (self.height() - h) / 2, w, h, True)

                y = self.height()*0.8
                v_align = Qt.AlignTop
            else:
                y = self.height() / 2
                if len(msgs) > 0:
                    y -= (len(msgs)-1)*fm.boundingRect(msgs[0]).height()*0.5
                v_align = Qt.AlignVCenter

            # WRITE TEXT
            for msg in msgs:
                fontw = fm.boundingRect(msg).width()
                self.render_text(self.width() / 2, y, msg, font, Qt.white, Qt.AlignHCenter, v_align)
                y += fm.boundingRect(msg).height()+5

        if self.STATUS == PresentationStatus.SUB_TRIAL_IN_PROGRESS:  # SUB-TRIAL
            self.display_sub_trial()

        if self.STATUS == PresentationStatus.STARTING_UP:  # INIT WINDOW
            font = QFont("Arial", min(self.width(), self.height())*0.05)
            font.setBold(True)
            fm = QFontMetrics(font)
            text = "Starting..."
            fontw = fm.boundingRect(text).width()
            fonth = fm.boundingRect(text).height()
            self.render_text(self.width() / 2, self.height() / 2, text, font, Qt.white, Qt.AlignHCenter,
                             Qt.AlignVCenter)

    def display_sub_trial(self):
        preserve_aspect = True

        col_count = self.width()/(self.cols*2)
        row_count = self.height()/(self.rows*2)
        col_w = self.width()/self.cols
        row_h = self.height()/self.rows

        i = 0
        #print(colored("Length of image list: %s" % len(self.imageOrder), 'cyan'))
        for im_str in self.imageOrder:
            im = self.images[im_str]

            col = i % self.cols
            row = math.floor(float(i)/float(self.cols))

            if self.MODE == PresentationMode.CIRCULAR_SEQUENCE_MODE:
                # get circular coordinates
                x_center, y_center = self.get_point_on_unit_sphere_from_radian(2*math.pi * (i / self.cols))
                circle_radius = int(min(self.width(), self.height()) / 2.5) # fine tuning
                x_center = (self.width()/2)  + (x_center * circle_radius)
                y_center = (self.height()/2) + (y_center * circle_radius)
            else:
                # get normal raster coordinates
                x_center = col_count + (col_w * col)
                y_center = row_count + (row_h * row)

            side = min(col_w, row_h) * 0.8

            if preserve_aspect:
                im_path = ''
                for k in self.images.keys():
                    if self.images[k] == im:
                        im_path = k
                        break
                im_dim = self.image_values[im_path]
                if im_dim.width() > im_dim.height():
                    w = side
                    h = int(float(side)*(float(im_dim.height())/float(im_dim.width())))
                else:
                    h = side
                    w = int(float(side)*(float(im_dim.width())/float(im_dim.height())))
            else:
                w = side
                h = w

            if self.MODE == PresentationMode.SPELLER_MODE:
                if self.HIGHLIGHT_COL:
                    highlight = col == self.TARGET_INDEX
                else:
                    highlight = row == self.TARGET_INDEX

            elif self.MODE == PresentationMode.SEQUENCE_MODE or self.MODE == PresentationMode.CIRCULAR_SEQUENCE_MODE:
                highlight = i == self.TARGET_INDEX
            else:
                # ERROR!!
                return

            self.display_image(im, x_center - (w / 2), y_center - (h / 2), w, h, highlight)
            i += 1

    @staticmethod
    def display_image(image, x, y, w, h, highlight):
        glBindTexture(GL_TEXTURE_2D, image)
        if highlight:
            glColor3f(1.0, 1.0, 1.0)
        else:
            glColor3f(0.2, 0.2, 0.2)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x+w, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x+w, y+h)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x, y+h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def load_images(self):
        self.images = dict()
        self.image_values = dict()

        for im_file in self.files:
            self.load_image(self.image_path + im_file)
            print("loaded %s from %s" % (im_file, self.image_path))
        shuffle_enabled = False
        if shuffle_enabled:
            shuffle(self.imageOrder)

        num_im = len(self.images)
        print("number loaded images " + str(num_im))

        # calculate number of rows for stim matrix in case of speller mode - otherwise set to 1
        if self.MODE == PresentationMode.SPELLER_MODE:
            self.rows = int(math.floor(math.sqrt(num_im)))
        else:
            self.rows = 1
        self.cols = int(math.ceil(float(num_im)/float(self.rows)))

    def load_image(self, path):
        qim = QImage(path).mirrored()
        qim = qim.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
        ptr = qim.bits()
        ptr.setsize(qim.byteCount())
        image_data = np.asarray(ptr).reshape(qim.width(), qim.height(), 4)

        im = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, im)
        glTexImage2D( GL_TEXTURE_2D, 0, 4, qim.width(), qim.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glDisable(GL_ALPHA_TEST)
        self.images[path] = im
        self.image_values[path] = QRect(0, 0, qim.width(), qim.height())
        self.imageOrder.append(path)

    def keyPressEvent(self, event):

        if self.quit_with_escape and event.key() == Qt.Key_Escape:
            self.close()

        if self.STATUS == PresentationStatus.TARGET_SHOW:
            self._key_pressed = True
            event.ignore()
            return
        elif self.STATUS == PresentationStatus.SUB_TRIAL_IN_PROGRESS \
                or self.STATUS == PresentationStatus.STARTING_UP:
            self._key_pressed = True
            event.ignore()
            return

        # END PROGRAM WITH ESCAPE IN IDLE MODE -> MATLAB MAY CRASH
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            event.ignore()
            self.close()

    def get_point_on_unit_sphere_from_radian(self, rad):
        return math.cos(rad), math.sin(rad)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    presentation = StimulusPresentation(False)
    # init speller-mode with image path and inter-stimulus-interval,
    presentation.initialize('speller_images/', 120, PresentationMode.SPELLER_MODE)
    presentation.show()
    presentation.show_target(-2, '', True)
    flash = presentation.execute_sub_trial()
    presentation.show_target(5, 'Search for this Target', True)
    flash = presentation.execute_sub_trial()
    presentation.show_target(12)
    presentation.show_target(-2, '', True)
    presentation.show_target(-1, 'This is the End.\nPress any key to exit.', skip_wait=True)
    sys.exit(app.exec_())
