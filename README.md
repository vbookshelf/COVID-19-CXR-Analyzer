## COVID-19-CXR-Analyzer
This flask web app uses a fine tuned Yolov5 model to detect and localize COVID-19 on chest x-rays.

This demo will be live until 31 August 2021.<br>
Demo App: https://covid19.test.woza.work/

<br>

<img src="http://covid19.test.woza.work/assets/covid-19-cxr-analyzer.png" width="700"></img>

<br>

The Yolov5 model that powers this app was fine tuned using data made available during the Kaggle SIIM-FISABIO-RSNA COVID-19 Detection competition. <br>
https://www.kaggle.com/c/siim-covid19-detection/overview

The dataset license info can be found here:<br>
https://www.kaggle.com/c/siim-covid19-detection/data


If you specifically want to see how the yolov5 model was deployed then please review the code in the views.py file.<br>
[Covid-Backend/flask/app/views.py]

I created a Kaggle notebook that explains how to use Yolov5. If you are a beginner I suggest reviewing that notebook before reviewing the views.py file.<br>
https://www.kaggle.com/vbookshelf/basics-of-yolo-v5-balloon-detection



### Notes

1- The frontend and backend code is available in this repo.<br>
2- The model was too large to be uploaded.<br>
3- Setup notes are included in a file called covid-app-notes.txt. This file is located inside the main folder.<br>
4- The .dockerignore file may not be visible. Please create this file if you don't see it.<br>
5- There is folder called 'static' containing a predict.html file. This folder is not essential and can be deleted.<br>

## References

1- Ultralytics Yolov5<br>
https://github.com/ultralytics/yolov5

2- The code is set up to be run as a Docker container. It's based on this video tutorial:<br>
Julian Nash docker and flask video tutorial<br>
https://www.youtube.com/watch?v=dVEjSmKFUVI

