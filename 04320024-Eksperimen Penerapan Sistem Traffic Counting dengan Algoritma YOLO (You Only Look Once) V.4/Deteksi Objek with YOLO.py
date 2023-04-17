# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Import the necessary packages
import cv2
import os
import re

# get working directory
loc = os.path.abspath('')
inputFile = loc+'/trafficCounter/inputs/625. 201709301058.mp4'

camera = re.match(r".*/(\d+)_.*", inputFile)
camera = camera.group(1)

vidcap = cv2.VideoCapture(inputFile)
success, image = vidcap.read()
count = 0;
while success:
    success, image = vidcap.read()

    # save frame as JPEG file
    cv2.imwrite(loc”/trafficCounter/outputs/"+camera+"_frame%d.jpg” % count, image)
    if cv2.waitKey(10) == 27:                    # exit if Escape is hit
        break
    count += 1

mask = np.zeros((frame_h,frame_w), np.uint8)
mask[:,:] = 255
mask[:120, :] = 0

frame_no = 0
ret, img = cap.read()
while ret:
    ret, ing = cap.read()
    frame_no = frame_no + 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply mask
    gray = cv2.bitwise_and(gray, gray, mask = mask)

    # image, reject levels level weights.
    cars = car_cascade.detecthultiscale(gray, 1.068, 5)

    # add this
    for (x,y,u,h) in cars:
        cv2.rectangle(img, (x,y), (x+u,y+h), (255,255,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    print( 'Processing %d : cars detected : [%s]' % (frame_no, len(cars)))

    cv2.imshou('img', img)
    if cv2.waitKey(27) and 0xFF == ord('q') : 
        break


cap.release()
cv2. destroyAllWindous ()

labelsPath = os.path.sep.join([args["yolo"], "data.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed (42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[@:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences. append(float(confidence))
            classIDs.append(classID)
