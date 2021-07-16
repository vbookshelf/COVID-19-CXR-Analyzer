## COVID-19-CXR-Analyzer - Under Construction
This flask web app uses a fine tuned Yolov5 model to detect and localize COVID-19 on chest x-rays.

This demo will be live until 31 August 2021. [ Still under construction ]<br>
Demo App: https://covid19.test.woza.work/

<br>

<img src="http://covid19.test.woza.work/assets/covid-19-cxr-analyzer.png" width="700"></img>

<br>

The Yolov5 model that powers this app was fine tuned using data made available during the Kaggle SIIM-FISABIO-RSNA COVID-19 Detection competition. <br>
https://www.kaggle.com/c/siim-covid19-detection/overview

Yolov5 repo:<br>
https://github.com/ultralytics/yolov5

The dataset license info can be found here:<br>
https://www.kaggle.com/c/siim-covid19-detection/data

The frontend and backend code is available in this repo. The model was too large to be uploaded. The model should be placed inside the folder called "yolov5".

The code is set up to be run as a Docker container. It's based on this video tutorial:

Julian Nash docker and flask video tutorial<br>
https://www.youtube.com/watch?v=dVEjSmKFUVI


The .dockerignore file may not be visible. Please create this file if you don't see it. In this repo I've included a txt file that explains the steps for installing Docker and Docker Compose on a Linux server. There is folder called 'static' containing a predict.html file. This folder is not essential and can be deleted. 

If you specifically want to see how the yolov5 model was deployed then please review the code in the views.py file.

I've created a Kaggle notebook that explains how to use Yolov5. If you are a beginner I suggest reviewing that notebook before reviewing the views.py file.<br>
https://www.kaggle.com/vbookshelf/basics-of-yolo-v5-balloon-detection
