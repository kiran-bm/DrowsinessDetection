#!/usr/bin/env python

import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.dates as mdates

import multiprocessing
from threading import Timer
import random
from numpy import ndarray

import sharedmem
from scipy.ndimage.interpolation import shift

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 7
MOUTH_AR_THRESH = 0.4

SHOW_POINTS_FACE = False
SHOW_CONVEX_HULL_FACE = False
SHOW_INFO = False

ARRAY_SIZE = 200

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
shared = sharedmem.empty(ARRAY_SIZE)
#shared[:] = np.random.rand(1, ARRAY_SIZE)[0]
print(shared)

sharedX = sharedmem.empty(ARRAY_SIZE)

def initXs():
	i = 1
	while True:
		sharedX[i-1] = i
		i+=1
		if i > ARRAY_SIZE:
			break
	print("Shared X", sharedX)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])	
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 

videoSteam = cv2.VideoCapture(0)
ret, frame = videoSteam.read()
size = frame.shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

focal_length = size[1]
center = (size[1]/2, size[0]/2)

camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

dist_coeffs = np.zeros((4,1))

t_end = time.time()

result = 0

#Create a queue to share data between process
q = multiprocessing.Queue()

def plot(i, xs, ys):

	global result
	global shared
	global sharedX
	
	try:
		result=q.get()
	except:
		result = result
	
	ear = result
	
	#print("{EAR} : ", ear)
	
	# Read temperature (Celsius) from TMP102
	temp_c = ear

    # Add x and y to lists
	xs.append(datetime.datetime.now().strftime('%H:%M:%S.%f'))
	ys.append(temp_c)

    # Limit x and y lists to 20 items
	xs = xs[-ARRAY_SIZE:]
	ys = ys[-ARRAY_SIZE:]
	shared = shared[-ARRAY_SIZE:]
	sharedX = sharedX[-ARRAY_SIZE:]

    # Draw x and y lists
	ax.clear()
	ax.plot(sharedX, shared)

	ax.grid(axis="x", color="green", alpha=.3, linewidth=2, linestyle=":")
	ax.grid(axis="y", color="black", alpha=.5, linewidth=.5)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
	fig.autofmt_xdate()

    # Format plot
	plt.xticks(rotation=60, ha='right')
	plt.subplots_adjust(bottom=0.30)
	plt.title('EAR Value over time')
	plt.ylabel('EAR')

Frames = 0

def simulation(q, ys):

	while True:

		global COUNTER_BLINK
		global COUNTER_MOUTH
		global COUNTER_FRAMES_EYE
		global COUNTER_FRAMES_MOUTH
		global t_end
		global ear
		global SHOW_POINTS_FACE
		global SHOW_CONVEX_HULL_FACE
		global SHOW_INFO
		global shared
		global sharedX
		global Frames

		ret, frame = videoSteam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)

		Frames += 1

		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			jaw = shape[48:61]

			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye) 
			ear = (leftEAR + rightEAR) / 2.0
			mar = mouth_aspect_ratio(jaw)

			image_points = np.array([
									(shape[30][0], shape[30][1]),
									(shape[8][0], shape[8][1]),
									(shape[36][0], shape[36][1]),
									(shape[45][0], shape[45][1]),
									(shape[48][0], shape[48][1]),
									(shape[54][0], shape[54][1])
									], dtype="double")


			(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
			(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

			if SHOW_POINTS_FACE:
				for p in image_points:
					cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

			p1 = (int(image_points[0][0]), int(image_points[0][1]))
			p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

			if SHOW_CONVEX_HULL_FACE: 
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				jawHull = cv2.convexHull(jaw)

				

				cv2.drawContours(frame, [leftEyeHull], 0, (255, 255, 255), 1)
				cv2.drawContours(frame, [rightEyeHull], 0, (255, 255, 255), 1)
				cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)
				cv2.line(frame, p1, p2, (255,255,255), 2)


			if p2[1] > p1[1]*1.5 or COUNTER_BLINK > 25 or COUNTER_MOUTH > 2:
				cv2.putText(frame, "Send Alert!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
			if ear < EYE_AR_THRESH:
				COUNTER_FRAMES_EYE += 1

				if COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES:
					cv2.putText(frame, "Sleeping Driver!", (200, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				if COUNTER_FRAMES_EYE > 2:
					COUNTER_BLINK += 1
				COUNTER_FRAMES_EYE = 0

			if mar >= MOUTH_AR_THRESH:
				COUNTER_FRAMES_MOUTH += 1
			else:
				if COUNTER_FRAMES_MOUTH > 5:
					COUNTER_MOUTH += 1

				COUNTER_FRAMES_MOUTH = 0

			if (time.time() - t_end) > 60:
				t_end = time.time()
				COUNTER_BLINK = 0
				COUNTER_MOUTH = 0

		if SHOW_INFO:
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 450),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, "Blinks: {}".format(COUNTER_BLINK), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, "Mouths: {}".format(COUNTER_MOUTH), (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		key = cv2.waitKey(1) & 0xFF

		q.put(ear)
		#shift(shared, -1, cval=ear)
		
		if Frames > ARRAY_SIZE:
			n =[]
			n[:] = shared[:]
			n = n[-(ARRAY_SIZE - 1):]
			n.append(ear)
			shared[:]=n[:]

		
			m =[]
			m[:] = sharedX[:]
			m = m[-(ARRAY_SIZE - 1):]
			m.append(Frames)
			sharedX[:]=m[:]
		
		#shift(n, -1, cval=ear)
			#print("Shared shifted", sharedX)
			
		else:
			shared[Frames - 1] = ear

		cv2.imshow("Output",frame)
		key = cv2.waitKey(1) & 0xFF

def main():
	#ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=20)
	#plt.show()
	
	initXs()
	
	simulate=multiprocessing.Process(None,simulation,args=(q,xs))
	simulate.start()

	ani = animation.FuncAnimation(fig, plot, fargs=(xs, ys), interval=20)
	plt.show()

	videoSteam.release()  
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
