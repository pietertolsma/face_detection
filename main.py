from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import face_recognition
import pickle
import logging
import threading



ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="face/encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

COLORS = np.random.uniform(0, 255, size=(20, 3))

# loop over the frames from the video stream
k = 0
frame = vs.read()
boxes = face_recognition.face_locations(frame, model=args["detection_method"])
encodings = face_recognition.face_encodings(frame, boxes)
names = []

time_in_image = {}

def draw_boxes(frame, args):
	global boxes
	global encodings
	global names
	boxes = face_recognition.face_locations(frame, model=args["detection_method"])
	encodings = face_recognition.face_encodings(frame, boxes)
	# loop over the facial embeddings
	names = []
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		# update the list of names
		names.append(name)

curr_time = time.time()

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face

	frame = vs.read()

	if k % 5 == 0:
		k = 0
		x = threading.Thread(target=draw_boxes, args=(frame, args))
		x.start()

		time_spent = time.time() - curr_time
		curr_time = time.time()
		for name in names:
			if name != "Unknown" and name in time_in_image:
				time_in_image[name] = time_in_image[name] + time_spent
			elif name != "Unknown":
				time_in_image[name] = time_spent
		print(time_in_image)

	# loop over the recognized faces
	i = 0
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom), COLORS[i], 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, COLORS[i], 2)
		i = i + 1

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
	k  = k + 1
