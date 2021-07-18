
# The original prediction code simulation is in this kernel:
# https://www.kaggle.com/vbookshelf/e-33-wheat-flask-app-inference-code

# Handling base64 images is simulated in this kernel:
# https://www.kaggle.com/vbookshelf/tb-my-flask-python-app-workflow

# This simulated code for this app is in this notebook:
# https://www.kaggle.com/vbookshelf/exp-134-covid-my-flask-yolopython-app-workflow

# *** NOTE: This entire file will get imported into __init__.py ***
# ------------------------------------------------------------------

#from the app folder import the app object
from app import app


from flask import request
from flask import jsonify
from flask import Flask
import base64
from PIL import Image
import io

import base64

import torch
import torchvision

import pandas as pd
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt



# Are we using and special packages?
# ------------------------------------

# [1] We are using an offline version of Yolov5
# that is in a folder called yolov5.

# ----end



# Config
# -------

YOLO_IMAGE_SIZE = 256
yolo_model_path_v5m = '~/Covid-Backend/flask/best.pt'



# -------------------------------
# Define the Model Architectures
# -------------------------------





# -------------------------------
# Define the helper functions
# -------------------------------
	
def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
# Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
	im = Image.fromarray(array)
	
	if keep_ratio:
		im.thumbnail((size, size), resample)
	else:
		im = im.resize((size, size), resample)
	
	return im
	

	
def image_to_data_url(image_path, image_extension):
	"""
	Image extension can be jpg, jpeg or png.
	
	"""
	ext = image_extension
	prefix = f'data:image/{ext};base64,'
	with open(image_path, 'rb') as f:
		img = f.read()
	return prefix + base64.b64encode(img).decode('utf-8')
		
		
		

# Draw the boxes on the image

def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=15):
	
	"""
	Set text=None to only draw a bbox without
	any text or text background.
	E.g. set text='Balloon' to write a 
	title above the bbox.
	
	Output: 
	Returns an image with one bounding box drawn.
	The title is optional.
	To draw a second bounding box pass the output image
	into this function again.
	
	"""
	
	
	w = xmax - xmin
	h = ymax - ymin
	
	# Draw the bounding box
	# ......................
	
	start_point = (xmin, ymin) 
	end_point = (xmax, ymax) 
	bbox_color = (255, 0, 0) 
	bbox_thickness = line_thickness
	
	image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness) 
	
	
	
	# Draw the tbackground behind the text and the text
	# .................................................
	
	# Only do this if text is not None.
	if text:
	    
		# Draw the background behind the text
		text_bground_color = (0,0,0) # black
		cv2.rectangle(image, (xmin, ymin-150), (xmin+w, ymin), text_bground_color, -1)
		
		# Draw the text
		text_color = (255, 255, 255) # white
		font = cv2.FONT_HERSHEY_DUPLEX
		origin = (xmin, ymin-30)
		fontScale = 3
		thickness = 10
		
		image = cv2.putText(image, text, origin, font, 
		                   fontScale, text_color, thickness, cv2.LINE_AA)
	
	
	
	return image
	
	
	
# -------------------------------
# RUN THE CODE
# -------------------------------

# Define the device
# ------------------
device = "cpu"


# Load the models
# ----------------



# Define the endpoints
# ---------------------


@app.route('/')
def index():
	return 'Hello world. I am a flask app.'
	
	

@app.route('/test')
def test():
	return 'Testing testing...'



# To access this endpoint navigate to:
# server_ip_address/static/predict.html
# Update the server ip address in the static/predict.html file.
# This endpoint has an html page in the static flask folder.

@app.route("/predict", methods=["POST"])
def predict():
    
	# receive the image that the user submitted
	message = request.get_json(force=True)
	base64Image = message['image']
	fname1 = message['image_fname']
	
	# create a folder to store the image
	#yolo_images_dir = '~/Covid-Backend/flask/yolo_images_dir'
	#os.mkdir(yolo_images_dir)
	
	# Decode the base64 image.
	# The result is a PIL image not a numpy array.
	decoded = base64.b64decode(base64Image)
	PIL_image = Image.open(io.BytesIO(decoded))
	
	
	# Convert the PIL image into a numpy array.
	np_image = np.array(PIL_image)
	
	# Resize the image
	pil_image = resize(np_image, size=YOLO_IMAGE_SIZE, keep_ratio=False, resample=Image.LANCZOS)
	
	
	# get the current working directory
	orig_dir = os.getcwd()
	
	
	# change the working directory
	os.chdir('yolov5')
	
	# Save the image in the folder
	# that we created.
	fname = 'user_image.jpg'
	#dst = os.path.join(yolo_images_dir, fname)
	pil_image.save('user_image.jpg')
	print('Image saved.')
	
	# Yolo Model:
	# Make a prediction on all images in images_dir
	# The model only creates a txt file if it finds objects on an image.
	test_images_path = 'user_image.jpg'
	
	# Execute a shell command to run Yolov5
	os.system('python detect.py --source "user_image.jpg" --weights "exp145-best.pt" --img 256 --save-txt --save-conf --exist-ok')
	
	print('Prediction completed.')
    
    # If the model has created a txt file for the image
	if os.path.exists('runs/detect/exp/labels/user_image.txt'):
		
		pred = 'Typical for COVID-19'
	    
		print('Processing predicted bboxes.')
		
		# This is how to put the contents of a txt
		# file into a dataframe.
		
		path = 'runs/detect/exp/labels/user_image.txt'
		
		# create a list of column names
		cols = ['class', 'x-center', 'y-center', 'bbox_width', 'bbox_height', 'conf-score']
		
		# put the file contents into a dataframe
		df_test_preds = pd.read_csv(path, sep=" ", header=None)
		
		# add the column names to the datafrae
		df_test_preds.columns = cols
		
		
		# Remember that Yolo preds are normalized.
		# Need to convert them into dimensions for the comp submission.
		# The dimensions that we submit need to be based on the original comp image sizes.
		
		orig_image_w = YOLO_IMAGE_SIZE
		orig_image_h = YOLO_IMAGE_SIZE
		
		df_test_preds['w'] = df_test_preds['bbox_width'] * orig_image_w
		df_test_preds['h'] = df_test_preds['bbox_height'] * orig_image_h
		
		df_test_preds['x_cent'] = orig_image_w * df_test_preds['x-center']
		df_test_preds['y_cent'] = orig_image_h * df_test_preds['y-center']
		
		df_test_preds['xmin'] = df_test_preds['x_cent'] - (df_test_preds['w']/2)
		df_test_preds['ymin'] = df_test_preds['y_cent'] - (df_test_preds['h']/2)
		
		df_test_preds['xmax'] = df_test_preds['xmin'] + df_test_preds['w']
		df_test_preds['ymax'] = df_test_preds['ymin'] + df_test_preds['h']
		
		
		# Read the image
		image = plt.imread('user_image.jpg')
	
	
	    # Draw the bboxes on the image
		for i in range(0, len(df_test_preds)):
	
			xmin = int(df_test_preds.loc[i, 'xmin'])
			ymin = int(df_test_preds.loc[i, 'ymin'])
			xmax = int(df_test_preds.loc[i, 'xmax'])
			ymax = int(df_test_preds.loc[i, 'ymax'])
			
			image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=2)
	
	
		# save the image
		dst = 'user_image.jpg'
		cv2.imwrite(dst, image)
		
		# convert the image to a base64 image
		image_path = 'user_image.jpg'
		image_extension='jpg'
		
		# Convert the image into a dataURL
		dataURL = image_to_data_url(image_path, image_extension)
		
		# Remove the prefix text that we identified above
		# This prefix will be put back just before the image is displayed 
		# on the page.
		base64Image = dataURL.replace("data:image/jpg;base64,", "")
		
		print('Bbox processing completed.')
	
	else:
		
		pred = 'Negative for pneumonia'
	    
		print('No bboxes detected.')
		
		# Send back the original image
		
		# convert the image to a base64 image
		image_path = 'user_image.jpg'
		image_extension='jpg'
		
		# Convert the image into a dataURL
		dataURL = image_to_data_url(image_path, image_extension)
		
		# Remove the prefix text that we identified above
		# This prefix will be put back just before the image is displayed 
		# on the page.
		base64Image = dataURL.replace("data:image/jpg;base64,", "")
	    
	    
	
	# Delete the exp folder
	# change the working directory
	print('Delete exp...')
	#os.chdir('yolov5') 
	if os.path.isdir('runs/detect/exp') == True:
		shutil.rmtree('runs/detect/exp')
	
		print('exp folder deleted.')
	
	
	# Change the working directory.
	# Go back one level
	#os.system('cd ..')
	os.chdir(orig_dir) 
	
	
	#pred = 777.55
	
	response = {
		'prediction': {
			'pred_class': pred,
			'image': base64Image,
			'image_fname': fname1,
		}
	}
	
	return jsonify(response)
	
	
	