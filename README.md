# Driver Assist Machine Learning Using Mask-RCNN

## What is Mask R-CNN:

<ul>
 	<li> R-CNN stands for "Regions with CNN features", CNN stands for "Convolutional Neural Network" </li>
  <li> R-CNN grabs parts of an image (or region) as a bounding box, and computes each region for CNN features, it then classifies each region to determine what it is through ROI align, testing pixel by pixel to form the mask, R-CNN then takes the output from the ROI align and helps generate the bounding boxes and classifies the target to determine what it is </li>
<li> mask R-CNN provides pixel level segmentation to mask over cars </li>
</ul>


<p align="center">
<strong> In this case road obstacles ahead are being masked: </strong>
<img src="https://github.com/shailahmed44/Driver-Assist-Machine-Learning-With-RCNN-Mask/blob/master/screenshots/footage.gif" width=800>
</p>


<hr>

## Installation and Setup:

<strong> When installing or running python programs: </strong>

  <h6> *Either "py / py -3.6 / python / python3", it varies on what you have installed and setup </h6>
  
<ul>
 
<li> Download Python 3.6:  </li>

> https://www.python.org/downloads/release/python-365/ 
 	
  
<li> Install and download cuda: </li>

> https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64 
 

<li>  Install and download Download <strong>cuDNN v7.6.5 for CUDA 9.0</strong> on Windows:  </li>
 
 <h6> *You will need to sign up for a developer nvidia account </h6> 
 
 > https://developer.nvidia.com/rdp/cudnn-archive 


<li> Copy the following files from the <strong> "cudnn-9.0-windows10-x64-v7.6.5.32.zip" </strong> rar file into the <strong>CUDA Toolkit directory. </strong></li>
<li> Copy <strong> <installpath>\cuda\bin\cudnn64_7.dll </strong> to <strong> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin. </strong></li>
<li> Copy <strong> <installpath>\cuda\ include\cudnn.h</strong>  to <strong> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include. </strong></li>
<li> Copy <strong> <installpath>\cuda\lib\x64\cudnn.lib </strong> to <strong> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64. </strong></li> 
 


 <li>  Copy <strong>requirements.txt</strong> from the root directory in the "RequirementsFile" folder and place it in <strong>"D:\Python3\Scripts"</strong> and install it: </li>
 
 > pip install tkintertable
 
 > pip install -r requirements.txt




 <li>  Change directory to <strong>"Driver-Assist-Machine-Learning-With-RCNN-Mask-master"</strong> folder, and install the dependencies:  </li>
 
> py -3.6 setup.py install



 <li> Download the weight <strong>(mask_rcnn_coco.h5)</strong> from the releases page, and move it to the <strong>"Driver-Assist-Machine-Learning-RCNN-Mask-master\mrcnn"</strong> folder: </li>
 
> https://github.com/matterport/Mask_RCNN/releases 
 

<hr>
<h6>  *Not Required since repo already includes it, but kept as reference: </h6> 

<li> Clone the cocoapi repo and extract into Driver-Assist-Machine-Learning-With-RCNN-Mask-master folder: </li> 

> https://github.com/philferriere/cocoapi


<hr>

 <li> cd into the extracted cocoapi folder and change to PythonAPI and install, path is <strong>"D:\Python3\Driver-Assist-Machine-Learning-With-RCNN-Mask-master\cocoapi-master\PythonAPI"</strong>: </li>

> py -3.6 setup.py build_ext install
 

<li> Replace the video source file in the <strong> "VideoSourceFile" </strong> folder and change the code in line 113 with the name of your own footage to apply the masking to pre-captured footage.  </li>

> Line 113:  stream = cv2.VideoCapture("VideoSourceFile/Freewaytest.mp4")

</ul>

<hr>

## Running the Application

If you are having issues running the demo.py, then delete the .egg file in <strong>"D:\Python3\lib\site-packages\mask_rcnn-2.1-py3.6.egg"</strong>.
It will be rebuilt when your run the applications.

<strong> Run the application: </strong>

<strong> *For pre-captured footage use the following below, this will create an output.mp4 with the masking and bounding box: </strong> 

> py -3.6 DAML_RCNN_Mask.py

<strong> *For real-time capture use the following below, bare in mind this is resource intensive: </strong> 

> py -3.6 DAML_RCNN_Mask_RealTime.py

<strong> If you have any other issues check the following link for other solutions: </strong>

> https://github.com/tensorflow/tensorflow/issues/36111

<hr>

## Getting Started with the Jupyter Notebook File (if you don't have it installed):

<strong> General Install through pip: </strong> 
>  pip install jupyterlab 
 
<strong> If conda: </strong> 
> conda install -c conda-forge notebook

<strong> Guide for installing notebook: </strong> 
https://jupyter.org/install

<hr>

## Opening Project Files with Jupyter Notebook:

<strong> cmd from "D:/python3/scripts", and run the following command to open jupyter notebook: </strong> 
> jupyter notebook

Copy the provided link, and paste in browser if it doesn't open on its own.

<hr>

## Source:

<strong> Mask R-CNN for Object Detection and Segmentation: </strong> 
<ul>
 	<li> Created by Matterport, Inc 	</li>
 	<li> Source: https://github.com/matterport/Mask_RCNN 	</li>
</ul>

<strong> COCO API: </strong> 
<ul>
 	<li> Cloned by github user philferriere 	</li>
 	<li> Source: https://github.com/philferriere/cocoapi 	</li>
</ul>



