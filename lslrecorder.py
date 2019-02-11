from pylsl import StreamInlet, resolve_stream
from threading import Thread
import numpy as np
from PyQt5.Qt import QMutex


class Lslrecorder:

    def __init__(self):
        self.mutex = QMutex()
        self.timeStamps = None
        self.stream = None
        self.inlet = None
        self.info = None
        self.channelCount = None
        self.doRec = False
        self.srate = None
        self.data = None
        self.bufferUpdateThread = None

    def findStream(self, hostname=None, timeout=1):
        # Gather lsl stream and create respective inlet and buffer, returns channelcount of that stream
        print("Searching for streams with a timeout of " + str(timeout) + " seconds")
        streams = resolve_stream(timeout)
        if len(streams) < 1:
            print("No stream found - exiting")
            return -1
        else:
            print("Found " + str(len(streams)) + " streams")
        if hostname is None:
            print("No stream hostname has been specified - selecting first stream")
            self.stream = streams[0]
        else:
            for stream in streams:
                if stream.hostname() == hostname:
                    self.stream = stream
                    print("Selected stream with hostname " + str(hostname))
                    break
            if self.stream is None:
                print("No stream with hostname " + str(hostname) + " has been found - exiting")

        self.inlet = StreamInlet(self.stream)
        self.info = self.inlet.info()
        self.channelCount = self.info.channel_count()
        self.srate = self.info.nominal_srate()
        self.data = np.empty((0, self.channelCount))
        try:
            self.offset = self.inlet.time_correction(timeout=3)
            print("Offset: " + str(self.offset))
        except TimeoutError:
            self.offset = 0
            print("Offset Retrieval Timed Out")

        #print("Stream Meta Info:")
        #print(self.info.as_xml())
        return self.channelCount

    def startRec(self):
        # Create and Start buffer update thread as daemon so that it gets terminated automatically when program exits
        self.doRec = True
        self.bufferUpdateThread = Thread(target=self.grabData, args=())
        self.bufferUpdateThread.daemon = True
        self.bufferUpdateThread.start()

    def stopRec(self):
        self.doRec = False
        self.bufferUpdateThread.join()
        print("Stopped recording")

    def grabData(self):
        print("Starting recording")
        while self.doRec:
            c, t = self.inlet.pull_chunk(timeout=0.0)
            if c:
                # add offset to timestamps and transform timestamps to ms and round to int
                tmp_t = np.array([int((ts + self.offset) * 1000) for ts in t])

                self.mutex.lock()
                self.data = np.concatenate((self.data, c), axis=0)
                if self.timeStamps is None:
                    self.timeStamps = np.array(tmp_t)
                else:
                    self.timeStamps = np.concatenate((self.timeStamps, tmp_t))
                self.mutex.unlock()

