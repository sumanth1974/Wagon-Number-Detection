from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import argparse
import pytesseract
import imutils
import os
from PIL import Image

tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


args={}
args["east"]='frozen_east_text_detection.pb'
args["min_confidence"]=0.5
args["width"]=320
args["height"]=320

def img_text(img_path):
	# load the image
	args["image"]=img_path
	image = cv2.imread(args["image"])

	# load the input image and grab the image dimensions

	try :
		orig = image.copy()
		(H, W) = image.shape[:2]
		print(args["width"], args["height"])
		# set the new width and height and then determine the ratio in change

		(newW, newH) = (args["width"], args["height"])
		rW = W / float(newW)
		rH = H / float(newH)

		# resize the image and grab the new image dimensions
		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		# define the two output layer names for the EAST detector model that
		# we are interested -- the first is the output probabilities and the
		# second can be used to derive the bounding box coordinates of text
		layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]

		# load the pre-trained EAST text detector
		print("[INFO] loading EAST text detector...")

		# print("layerNames")
		net = cv2.dnn.readNet(args["east"])

		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
									 (123.68, 116.78, 103.94), swapRB=True, crop=False)
		start = time.time()
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		end = time.time()

		# show timing information on text prediction
		print("[INFO] text detection took {:.6f} seconds".format(end - start))

		# grab the number of rows and columns from the scores volume, then
		# initialize our set of bounding box rectangles and corresponding
		# confidence scores
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []

		# loop over the number of rows
		for y in range(0, numRows):
			# extract the scores (probabilities), followed by the geometrical
			# data used to derive potential bounding box coordinates that
			# surround text
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]

			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have sufficient probability, ignore it
				if scoresData[x] < args["min_confidence"]:
					continue

				# compute the offset factor as our resulting feature maps will
				# be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)

				# extract the rotation angle for the prediction and then
				# compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				# use the geometry volume to derive the width and height of
				# the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]

				# compute both the starting and ending (x, y)-coordinates for
				# the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)

				# add the bounding box coordinates and probability score to
				# our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		i = 0

		# gray out the image
		orig1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

		# image blurring
		#orig1 = cv2.blur(orig1, (1, 1))

		# threshold & invert
		#ret, thresh = cv2.threshold(orig1, 107, 255, cv2.THRESH_BINARY_INV)
		#thresh_copy = thresh.copy()

		#kernel1 = np.ones((3, 3), np.uint8)
		#img_erosion = cv2.erode(thresh, kernel1, iterations=1)

		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			# print(startX,startY,endX,endY)
			# draw the bounding box on the image
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

			crop_img = orig1[startY + 2:endY + 90, startX - 50:endX + 20]
			cv2.imwrite("detected/detected%d.jpg" % i, crop_img)
			i = i + 1

		f = open("output.txt", "a+")

		try:

			for j in range(0, i):

				end = "detected" + str(j) + ".jpg"

				filename2 = "C:\\Users\\nehul\\Downloads\\opencv-text-detection\\opencv-text-detection\\detected\\" + end
				filename3= cv2.imread(filename2)
				width1, height1,channels = filename3.shape
				#print(width1,height1)
				resized_img = cv2.resize(filename3, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
				blur = cv2.blur(resized_img,(2,2))
				#cv2.waitKey(0)

				#threshold & invert
				ret, thresh = cv2.threshold(blur, 120, 230, cv2.THRESH_BINARY_INV)
				thresh_copy = thresh.copy()

				kernel1 = np.ones((3,3), np.uint8)
				img_erosion = cv2.erode(thresh, kernel1, iterations=1)

				#cv2.imshow("Show",img_erosion.copy())
				#cv2.waitKey(0)

				text = pytesseract.image_to_string(img_erosion.copy(), config='--psm 6', )

				print(j, text)

				if len(text) >= 9:
					f.write("Text in this line is %s\n" % text)


		except OSError as e:

			print("The error raised is", e)

		f.close()

	except AttributeError as a:

		print("The error is",a)

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        cv2.imwrite("frame%d.jpg" % count, image)
        img_text('frame%d.jpg' %count)
        count += 1

if __name__ == '__main__':
    FrameCapture("6.mp4")
