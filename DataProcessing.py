import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import io
import cv2
from PIL import Image
import requests
from io import BytesIO
import random
from tqdm import tqdm

base_path = 'data/raw'
images_boxable_fname = 'train-images-boxable.csv'
annotations_bbox_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'

images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
# print (images_boxable.head(5))
annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
# print (annotations_bbox.head(5))
class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname),names=["name", "class"])
# print(class_descriptions.head(5))

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

	# io.imsave('sample.jpg', img_bbox)

def shuffle_ids_perClass(class_name, n=1000):
	pd = class_descriptions[class_descriptions['class']==class_name]
	label_name = pd['name'].values[0]
	bbox = annotations_bbox[annotations_bbox['LabelName']==label_name]
	print('There are {} {} in the dataset'.format(len(bbox),class_name))

	img_ids = bbox['ImageID']
	img_ids = np.unique(img_ids) # one image can contain multiple objects

	# Shuffle the ids and pick the first 1000 ids
	copy_img_ids = img_ids.copy()
	random.seed(1)
	random.shuffle(copy_img_ids)
	shuffled_img_ids = copy_img_ids[:n]
	return shuffled_img_ids

def download_images(image_ids, class_name):
	for img_id in tqdm(image_ids):
		img_name = img_id+'.jpg'
		# print('processing {}'.format(img_name))
		img_url = images_boxable[images_boxable['image_name']==img_name]['image_url'].values[0]
		response = requests.get(img_url)
		img = Image.open(BytesIO(response.content))
		# img = np.array(img)
		# try:
		# 	io.imsave('data/images/'+class_name+'/'+img_name, img)
		# except Exception as e:
		# 	print(e, img_url)
		# 	img = img.convert('RGB')
		# 	io.imsave('data/images/'+class_name+'/'+img_name, img)
		img.save('data/images/'+class_name+'/'+img_name)

def create_train_csv():
	
	train_csv_df = pd.DataFrame(columns=['ID','FileName','ClassName','XMin','XMax','YMin','YMax'])
	base_path = 'data/images'
	for file_name in tqdm(os.listdir(base_path)):
		if not file_name.startswith('.'):
			for img_name in os.listdir(os.path.join(base_path,file_name)):
				if not img_name.startswith('.'):
					# print('processing {}'.format(img_name))
					img_id = img_name.strip('.jpg')
					rows = annotations_bbox[annotations_bbox['ImageID']==img_id]
					for _, row in rows.iterrows():
						xmin = row['XMin']
						xmax = row['XMax']
						ymin = row['YMin']
						ymax = row['YMax']
						label_name = row['LabelName']
						class_series = class_descriptions[class_descriptions['name']==label_name]
						class_name = class_series['class'].values[0]
						train_csv_df = train_csv_df.append({'ID':img_id,'FileName':img_name,'ClassName':class_name,'XMin':xmin,'XMax':xmax,'YMin':ymin,'YMax':ymax},ignore_index=True)

	train_csv_df.to_csv('data/train_csv.csv',index=False)

		


if __name__ == '__main__':
	# ids = shuffle_ids_perClass('Person')
	# download_images(ids,'Person')
	# ids = shuffle_ids_perClass('Mobile phone')
	# download_images(ids,'Mobile phone')
	# ids = shuffle_ids_perClass('Car')
	# download_images(ids,'Car')
	create_train_csv()
	
