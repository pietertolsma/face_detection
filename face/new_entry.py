import os
from imutils.video import VideoStream
from imutils.video import FPS
import time
import cv2

print("New Entry!")
name = input("Enter name: ")
path = "dataset/" + name

exists = os.path.isdir(path)

if exists:
    print("Adding pictures to existing entry")
else:
    print("Creating new entry")
    os.mkdir(path)


vs = VideoStream(src=0).start()

for i in range(0, 5):
    img = vs.read()
    print("Snap!")

    img_path = path + "/" + str(time.time()) + ".png"
    cv2.imwrite(img_path, img)

    time.sleep(1)

print("Done!")
