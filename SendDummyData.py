import time
from random import random as rand
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from time import time, sleep
from uptime import uptime

rate = 256
info = StreamInfo('Some Waves', 'EEG', 16, rate, 'float32', 'myuid34234')



timestampinms = time() * 1000
uptimestampims = uptime() * 1000
offsetinms = timestampinms - uptimestampims


print(timestampinms)
print(uptimestampims)
print(offsetinms)
timestampinms = round(timestampinms)
uptimestampims = round(uptimestampims)
offsetinms = round(offsetinms)
print(timestampinms)
print(uptimestampims)
print(offsetinms)

info.desc().append_child("synchronization").append_child_value("offset_mean",str(offsetinms))
# next make an outlet
outlet = StreamOutlet(info)

x = 0

print("now sending data...")
while True:
    mysample = [np.sin(x), np.cos(x), np.sin(x*2), np.cos(x*2), np.sin(x*4), np.cos(x*4), np.sin(x*8),
                np.cos(x*8), np.sin(x), np.cos(x), np.sin(x*2), np.cos(x*2), np.sin(x*4), np.cos(x*4), np.sin(x*8), np.cos(x*8)]

    outlet.push_sample(mysample)
    x += 1/rate
    sleep(1/rate)

