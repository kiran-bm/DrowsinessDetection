#!/usr/bin/env python

import cv2
import time
#import numpy as np
#from scipy.spatial import distance as dist
#from imutils import face_utils
#import dlib
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.dates as mdates

import multiprocessing
from threading import Timer
import random


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 7
MOUTH_AR_THRESH = 0.4

SHOW_POINTS_FACE = False
SHOW_CONVEX_HULL_FACE = False
SHOW_INFO = False

ear = 0
mar = 0

COUNTER_FRAMES_EYE = 0
COUNTER_FRAMES_MOUTH = 0
COUNTER_BLINK = 0
COUNTER_MOUTH = 0

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

videoSteam = cv2.VideoCapture(0)
ret, frame = videoSteam.read()

result = 0

def videoCapture(q):

	global result

	ret, frame = videoSteam.read()
	
	try:
		result=q.get_nowait()
	except:
		result = result
	
	ear = result
	
	q.put(ear)
	
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
	
	cv2.imshow("Output",frame)

	key = cv2.waitKey(1) & 0xFF

def simulation(q):
    iterations = range(100)
    for i in iterations:
        if not i % 10:
            try:
                print(q.get_nowait())
            except:
                print('0')

            time.sleep(1)
                #here send any data you want to send to the other process, can be any pickable object
            q.put(random.randint(1,10))

def main():
	#ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=20)
	#plt.show()
	
	#Create a queue to share data between process
	q = multiprocessing.Queue()

	simulate=multiprocessing.Process(None,simulation,args=(q,))
	simulate.start()

	while True:
		videoCapture(q)
		time.sleep(0.2)
	videoSteam.release()  
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()