import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import io
import cv2
from PIL import Image
import requests
from io import BytesIO

base_path = 'data'
images_boxable_fname = 'train-images-boxable.csv'
annotations_bbox_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'

images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
# print (images_boxable.head(5))
annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
# print (annotations_bbox.head(5))
class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname))
print(class_descriptions.head(5))


def readOneImg():
	image_name = images_boxable['image_name'][0]
	image_url = images_boxable['image_url'][0]
	# print(image_url)
	# img = io.imread(image_url)
	# with urllib.request.urlopen(image_url) as url:
	# 	arr = np.asarray(bytearray(url.read()), dtype=np.uint8)
	# 	print(url.read())
	# 	img = cv2.imdecode(arr, -1) # 'Load it as it is'
	response = requests.get(image_url)
	img = Image.open(BytesIO(response.content))
	return np.array(img), image_name.strip('.jpg')

def plot(img):
	# cv2.imshow('lalala', img)
	plt.imshow(img)
	plt.show()

def plotBbx(img, img_id):
	height, width, _ = img.shape
	plt.figure(figsize=(15,10))
	plt.subplot(1,2,1)
	plt.title('Original Image')
	plt.imshow(img)
	img_bbox = img.copy()
	bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
	# print(img_id)
	# print(row)
	for index, row in bboxs.iterrows():
		
		xmin = row['XMin']
		xmax = row['XMax']
		ymin = row['YMin']
		ymax = row['YMax']
		xmin = int(xmin*width)
		xmax = int(xmax*width)
		ymin = int(ymin*height)
		ymax = int(ymax*height)
		label_name = row['LabelName']
		class_series = class_descriptions[class_descriptions['name']==label_name]
		class_name = class_series['class'].values[0]
		cv2.rectangle(img_bbox,(xmin,ymin),(xmax,ymax),(0,255,0),2)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img_bbox,class_name,(xmin,ymin-10), font, 1,(0,255,0),2)

	plt.subplot(1,2,2)
	plt.title('Image with Bounding Box')
	plt.imshow(img_bbox)
	plt.show()

	io.imsave('sample.jpg', img_bbox)


if __name__ == '__main__':
	img, image_id=readOneImg()
	plotBbx(img, image_id)
