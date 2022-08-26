# USAGE
#python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

#python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

#python recognize_patent.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image images/adrian.jpg



# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

saveImagePath= 'SavedImages'

# print text
def plotimagemarking(img, name,pred,boxTuple):

	(startX,startY,endX,endY) = boxTuple
	scale = 240/624
	fontScale =1
	fontThickness= 4

	cv2.rectangle(image, (startX - 20, startY - 30), (endX + 25, endY + 25), (0, 255, 0), 2)
	cv2.putText(image, name.upper(), (startX-65, startY-40), cv2.FONT_HERSHEY_SIMPLEX, fontScale - (fontScale*scale), (0, 255, 0), int(fontThickness - (fontThickness*scale)))
	cv2.putText(image, pred, (endX - 50, endY - 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale - (fontScale*scale), (0, 255, 0), int(fontThickness - (fontThickness*scale)))

	#Crowd Medical Data that should be retrived from the medical watch
	lbls = ['Temp : {:.1f}C'.format(random.uniform(36.9,38.1)), 'BPM  : {}'.format(random.randint(40,110)), 'BP   : {}/{}'.format(random.randint(100,160),random.randint(60,105)), 'SaO2 : {}%'.format(random.randint(89,99))]
	offset = 35

	if (scale <0.8):
		offsetX = int( 135 - (95 * scale))
		offsetY = int(30 - (30 * scale))


	x = int(startX - offsetX)
	y = int(startY + offsetY)

	print ("rectangel height {}".format(endY + 25))

	for idx, lbl in enumerate(lbls):
		cv2.putText(image, str(lbl), (x, y + int(offset - (offset*scale)) * idx), cv2.FONT_HERSHEY_SIMPLEX,fontScale - (fontScale*scale), (255, 255, 255), int(fontThickness - (fontThickness*scale)))


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]


# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()


# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		#check top predications
		ind = np.argpartition(preds, -4)[-4:]
		print("Total preds {}".format(len(preds)))
		print(ind)
		print(preds[ind])
		print(le.classes_[ind])

		# draw the bounding box of the face along with the associated
		# probability
		person_info = "{} - {:.1f}%".format("Hajji: "+name, proba * 100)

        #box co-ordinates tuple
		y = startY - 10 if startY - 10 > 10 else startY + 10
		t=(startX, startY, endX, endY)
		plotimagemarking(image,"Hajji: {}".format(name), "{:.1f}%".format(proba*100),t)

		print(args["image"])
		cv2.imwrite(f'{saveImagePath}/{args["image"]}', image)
		# y = startY - 10 if startY - 10 > 10 else startY + 10
		#
		# cv2.rectangle(image, (startX-20, startY-30), (endX+25, endY+25),
		# 	(0, 255, 0), 2)
		#
		# frame = np.ones([400, 400, 3]) * 255
		# lbls = ['standUp', 'front', 'lookU', 'lookF', 'lookDF', 'HandOnHipR']
		#
		# offset = 35
		# x, y = 50,50
		#
	    # #cv2.putText(image, '[\'' + str(lbls[0]) + '\',', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		#
	    # for idx, lbl in enumerate(lbls):
	    #      cv2.putText(image, str(lbl), (x, y + offset * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )

	    #cv2.putText(frame, '\'' + str(lbls[0]) + '\']', (x, y + offset * (idx + 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
		#cv2.putText(image, person_info, (startX-20, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		#cv2.putText(image, health_info, (startX-40, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)


#more images for testing
#https://www.facebook.com/eng.ahmed.refaat/photos
#https://www.facebook.com/photo/?fbid=10151280251165470&set=bc.Abqzx02CnV5X3vgG5nRcwI71Vvd3c-xBmna9Eb8IsAzafdcT_oyKvhh4-sQ_ewlE-6Y7CeNqFvIT9WltlMrL-DE6aAE47VQcB8eybUSkbdNdQRx1fiiTJsQkbnjPHrdtwTTKNeC2Nz5XYhy1pA2XFqO9&opaqueCursor=Abo6OuzXEhPOcn05Y2OXe74v0qtQtvx-yY3VbxmxgFoaHm46WLYN9xkYh9cHAq1Rgy9c2Idih4kb7gxxmAUHEXp3ywhQiOhUSq-1cur5ldz7mbudlkhXBWk0PR5o-uLeVSHu-oSFbY4kao49HykHKT8bCy1ZiPMd-whwMjhHUBnH1lcpl4mAQYyMdhg08LAM2OVSXt72eRZ664vd9WzVhhCb8mwZeRobplmv0QF4xFs2T4h_YJgn2qbCbnnzYgdqqwqiDuPhNoYi6bniDeM-wQtWQt3ceeJK4Ml5SIH3e14F_DcP1xT1CWDx72UKXUUI_s27IZGgyD2Y6tlSvjRpayv2Mocl1CA5QBS_M9xOPKiJKBbOjzlCzg_cU8cHT7qXaIppYYehg2hhuUp60kXLZzfTtB7QUUYDgFI3A_LA_djd9ZnG00hHn3TURkLsXLzUIEmFoBV4pMWDRGR-or9PMnW067MedCPMurpKGdonrGxIbLotHo5Word_Pn5RGP2bid4
# mullti people image  https://www.facebook.com/photo/?fbid=10163710422705344&set=gm.2857141957899648
