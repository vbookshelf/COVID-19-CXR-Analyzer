
# This prediction code is simulated in this kernel:
# https://www.kaggle.com/vbookshelf/e-33-wheat-flask-app-inference-code

# Handling base64 images is simulated in this kernel:
# https://www.kaggle.com/vbookshelf/tb-my-flask-python-app-workflow

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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models # reg_model comes from here


import numpy as np
import cv2



# Special packages

# seg_model comes from here
import segmentation_models_pytorch as smp 
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# this is for the pre-processing
import albumentations as albu 
from albumentations import Compose



# Are we using and special packages?
# ------------------------------------

# Check if these are pre-intalled on PythonAnywhere.
# If not, these packages have to be pip installed onto the server:

# (1) Segmentation Models Pytorch
# https://github.com/qubvel/segmentation_models.pytorch
# $ pip install segmentation-models-pytorch

# (2) Albumentations (used to pre-process the image)
# https://github.com/albumentations-team/albumentations
# $ pip install albumentations

# ----end




# -------------------------------
# Define the Model Architectures
# -------------------------------





# -------------------------------
# Define the helper functions
# -------------------------------
	

	
	

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
	message = request.get_json(force=True)
	base64Image = message['image']
	#decoded = base64.b64decode(encoded)
	#image = Image.open(io.BytesIO(decoded))
	
	
	
	pred = 777.55
	
	
	response = {
	    'prediction': {
	        'wheat_count': pred,
			'image': base64Image,
	    }
	}
	return jsonify(response)
	
	
	