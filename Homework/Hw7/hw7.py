import cv2
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D as axes3d 

import glob
import h5py



    

def main():

	im = cv2.imread('Bean.jpg', 1);


	row, col = im.shape[:2]

	afMat = np.array([

		[0.02,  0.01, 10],
		[0.01, -0.02,  5]
		])

	base = np.array([
		[1, 0, 0],
		[0, 1, 0] 
		])

	sum = afMat + base;

	imWarp = cv2.warpAffine(im, sum, (col, row));

	imDiff = cv2.subtract(imWarp, im);

	cv2.imshow("Original", im);
	cv2.imshow("Warped", imWarp);
	cv2.imshow("Difference", imDiff);

	while True:
		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break

	cv2.destroyAllWindows()

	print('Part 3')


	# Chose 4 classes:
	# - Chihuahua
	# - Bagels
	# - Microwave
	# - Canoe

	# Problem 3
	from keras.applications.resnet50 import ResNet50
	from keras.preprocessing import image
	from keras.applications.resnet50 import preprocess_input, decode_predictions

	model = ResNet50(
	    include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

	# to get a lot of images at once
	images = glob.glob('photos/*.jpg')

	print('\n')

	for fname in images:

	    img_path = fname
	    img = image.load_img(img_path, target_size=(224, 224))
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x)

	    pred = model.predict(x)

	    label = decode_predictions(pred)[0][0][1]
	    confidence = decode_predictions(pred)[0][0][2]

	    confidence = round(float(confidence), 5)

	    print("Label: {}, Confidence: {}%".format(label, (confidence * 100)))

	    show = cv2.imread(fname, 1)

	    size = cv2.getTextSize("Label: {}".format(
	        label), 0, 0.5, 2)[0]

	    cv2.rectangle(show, (0, 0), (size[0] + 5,
	                                 size[1] + 10), (255, 255, 255), -1)
	    cv2.putText(show, "Label: {}".format(label), (5, 15),
	                0, 0.5, (0, 0, 0), 2, 4)

	    cv2.imshow("Classification", show)
	    cv2.waitKey(1000)
	cv2.destroyAllWindows()

	



if __name__ == "__main__":
    main()