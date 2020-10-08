# Driver Assist Machine Learning Using Mask-RCNN

## What is Mask R-CNN:

-R-CNN stands for "Regions with CNN features", CNN stands for "Convolutional Neural Network"
-R-CNN grabs parts of an image (or region) as a bounding box, and computes each region for CNN features,
it then classifies each region to determine what it is through ROI align, testing pixel by pixel to form the mask, 
R-CNN then takes the output from the ROI align and helps generate the bounding boxes and classifies the target to determine what it is
-mask R-CNN provides pixel level segmentation to mask over cars

In this case road obstacles ahead are being masked.

<hr>
## Installation and Setup:

-When installing or running python programs:
*Either "py / py -3.6 / python / python3", it varies on what you have installed and setup

Download Python 3.6:
https://www.python.org/downloads/release/python-365/

Install and download cuda:
https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64

Install and download Download cuDNN v7.6.5 for CUDA 9.0 on Windows: 
-you will need to sign up for a developer nvidia account
https://developer.nvidia.com/rdp/cudnn-archive

Copy the following files from the "cudnn-9.0-windows10-x64-v7.6.5.32.zip" rar file into the CUDA Toolkit directory.
Copy <installpath>\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin.
Copy <installpath>\cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include.
Copy <installpath>\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64.

Copy requirements.txt from the root directory in the "RequirementsFile" folder and place it in "D:\Python3\Scripts" and install it:
pip install tkintertable
pip install -r requirements.txt

Change directory to "Driver-Assist-Machine-Learning-With-RCNN-Mask-master" folder, and install the dependencies:
py -3.6 setup.py install

Download the weights (mask_rcnn_coco.h5) from the releases page, and move it to the "Driver-Assist-Machine-Learning-RCNN-Mask-master\mrcnn" folder:
https://github.com/matterport/Mask_RCNN/releases

<hr>
*Not Required since repo already includes it, but kept as reference:

Clone the cocoapi repo and extract into Mask_RCNN-master folder:
https://github.com/philferriere/cocoapi

<hr>

cd into the extracted cocoapi folder and change to PythonAPI and install, path is "D:\Python3\Driver-Assist-Machine-Learning-With-RCNN-Mask-master\cocoapi-master\PythonAPI":
py -3.6 setup.py build_ext install

If you are having issues running the demo.py, then delete the .egg file in "D:\Python3\lib\site-packages\mask_rcnn-2.1-py3.6.egg".
It will be rebuilt when your run the applications.

Replace the video source file in the "VideoSourceFile" folder and change the code in line 113 with the name of your own footage to apply the masking to pre-captured footage.
Line 113:  stream = cv2.VideoCapture("VideoSourceFile/Freewaytest.mp4")

Run the application:
*For pre-captured footage its the following below, this will create an output.mp4 with the masking and bounding box:
py -3.6 DAML_RCNN_Mask.py

*For real-time capture its the following below, bare in mind this is resource intensive:
py -3.6 DAML_RCNN_Mask_RealTime.py

If you have any other issues check the following link for other solutions:
https://github.com/tensorflow/tensorflow/issues/36111

<hr>
## Getting started with the Jupyter Notebook file (if you don't have it installed):
pip install jupyterlab

If conda:
conda install -c conda-forge notebook

Guide for installing notebook:
https://jupyter.org/install

<hr>

## Opening project files jupyter notebook:
cmd from D:/python3/scripts
> jupyter notebook

Copy the provided link, and paste in browser if it doesn't open on its own.

<hr>
## Source:

https://github.com/matterport/Mask_RCNN
https://github.com/philferriere/cocoapi
