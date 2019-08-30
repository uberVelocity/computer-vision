
# Computer Vision MVP 001: rotating equipment abnormal vibration detection, using computer vision
# Object detection based on colour range

# Test 001.02: OpenCV Track Object Movement with Edge Detection
# @2019 Phi Project / Ovidiu Bradin 
# based on PyImageSearch tutorial (OpenCV Track Object Movement) combined with Edge Detection (sobel_and_laplacian & canny)

# works with live camera or recorded video
# video file must be stored in the same folder
# video file name and extension must be passed via input arguments
# abnormal thresholds passed via input arguments

# Notes on using "#" marks:
# original reference
# .. author(s) contributions 


# USAGE
# python3 object_movement.py
# or
# python3 object_movement.py --video test_file_name.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
#.. d/delta = threshold for abnormal vibration
ap.add_argument("-d", "--delta", type=int, default=10,
	help="threshold for abnormal vibration")
args = vars(ap.parse_args())
	
# .. define the lower and upper boundaries of the "target"
# .. in the HSV color space
targetLower = (80, 121, 53)
targetUpper = (110, 201, 255)
 
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

delta_radius = 0

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
 
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

#.. initialize traffic light colours
tl_b = 0
tl_g = 255
tl_r = 0			
  
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()
 
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=800)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
	#.. convert frame to gray colour
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#.. Compute the Laplacian of the image
	lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
	lap = np.uint8(np.absolute(lap))
	#.. Compute Sobel gradient images
	sobelX = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0)
	sobelY = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1)

	sobelX = np.uint8(np.absolute(sobelX))
	sobelY = np.uint8(np.absolute(sobelY))

	sobelCombined = cv2.bitwise_or(sobelX, sobelY)

	#.. Canny edge detection
	canny = cv2.Canny(blurred, 50, 100)
	#canny = imutils.auto_canny(blurred)

	# construct a mask for the color "target", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, targetLower, targetUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
 
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	#.. reset traffic light colours
	tl_b = 0
	tl_g = 255
	tl_r = 0

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			pts.appendleft(center)

	# loop over the set of tracked points
	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")
	
			# ensure there is significant movement in the
			# x-direction
			# if np.abs(dX) > 20:
			# .. x-direction resolution changed as input parameter
			# "delta" (default=10)
			if np.abs(dX) > args["delta"]:
				dirX = "Right" if np.sign(dX) == 1 else "Left"
			#.. set traffic light colours
				tl_r = 255
			if np.abs(dX) > args["delta"]*3:
				tl_g = 0
	 
			# ensure there is significant movement in the
			# y-direction
			# if np.abs(dY) > 20:
			# .. y-direction resolution changed as input parameter
			# "delta" (default=10)
			if np.abs(dY) > args["delta"]:
				dirY = "Up" if np.sign(dY) == 1 else "Down"
			#.. set traffic light colours
				tl_r = 255
			if np.abs(dY) > args["delta"]*3:
				tl_g = 0
 
			# handle when both directions are non-empty
			if dirX != "" and dirY != "":
				direction = "{}-{}".format(dirY, dirX)
 
			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (255, 243, 211), thickness)
 
	# show the movement deltas and the direction of movement on
	# the frame
	cv2.putText(frame, direction, (25, 30), cv2.FONT_HERSHEY_COMPLEX,
		0.65, (122, 74, 18), 2)
	#.. traffic light
	cv2.putText(frame, "o", (5, 28), cv2.FONT_HERSHEY_COMPLEX,
		0.65, (tl_b, tl_g, tl_r), 4)

	#.. show also radius
	cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY) + " / d: " + str(int(2*radius)),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (255, 232, 157), 1)
 
	# show the frame to our screen and increment the frame counter
	cv2.imshow("Phi Project: Vibration Detection Test", frame)

	#.. show also canny image (smaller window, under the main window)
	cv2.imshow("Canny", cv2.resize(canny, (400,300)))
	cv2.moveWindow("Canny", 50, 660)

	#.. show also the sobel image (smaller window, under the main window)
	cv2.imshow("Sobel", cv2.resize(sobelCombined, (400,300)))
	cv2.moveWindow("Sobel", 470, 660)

	#.. show also the laplacian image (smaller window, under the main window)
	#cv2.imshow("Laplacian", cv2.resize(lap, (400,300)))
	#cv2.moveWindow("Laplacian", 50, 660)

	key = cv2.waitKey(1) & 0xFF
	counter += 1
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the camera
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()

