#!/usr/bin/env python3

#/usr/bin/env conda run -n tensorflow_1_14 python

#usage
#python time_sprint.py <filename> <manual_fps (optional)>

#filename: name of video file in the input folder. 
#manual_fps (optional): set the fps, if for example using an editted slowmo video where the metadata fps will give the frames per video seconds, but not actual seconds

#Remember to activate conda environment with correct package versions
#activate tensorflow_1_14



#Packages
import os
import sys
import shutil
#import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import json
#import time
import imageio
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


#Produce error if no filename given as argument
if len(sys.argv) < 2:
    print('Not enough arguments given... \n Should run as time_sprint.py <filename> <manual_fps (optional)>')
    sys.exit()

#video filename
fname = sys.argv[1]

#Manual fps
if len(sys.argv)>2:
    manual_fps = sys.argv[2]


#---------------------------------------------------------------------------------------------#
# Configurations
#---------------------------------------------------------------------------------------------#

#Import Mask RCNN (cloned repo from https://github.com/matterport/Mask_RCNN.git)
mask_rcnn_path = "../../../../cloned_repos/Mask_RCNN/"
print(mask_rcnn_path)
sys.path.append(mask_rcnn_path)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Import COCO config
sys.path.append(os.path.join(mask_rcnn_path, "samples/coco/"))  
import coco

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

#model location
models_dir = "../models/"


#---------------------------------------------#
#Person model

#Pre-trained COCO weights
coco_mod = os.path.join(models_dir, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(coco_mod):
    utils.download_trained_weights(coco_mod)


#Define model
person_model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
person_model.load_weights(models_dir+'mask_rcnn_coco.h5', by_name=True)

# COCO Class names
coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


#---------------------------------------------#
#Cones model

#config
class ConesConfig(Config):
    """Configuration for training on the orange cone dataset.
    Derives from the base Config class and overrides values specific
    to the cones dataset.
    """
    # Give the configuration a recognizable name
    NAME = "orange_cones"
    
    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (cone)
    
    # All of our training images are 1080x1920 (video captures from videos taken on phone)
    IMAGE_MIN_DIM = 1080
    IMAGE_MAX_DIM = 1920
    
    # Steps per epoch
    #STEPS_PER_EPOCH = 500
    STEPS_PER_EPOCH = 20 
    
    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, this is downsized
    BACKBONE = 'resnet50'
    
    # Other specifications
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    

class InferenceConfig2(ConesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    
inference_config2 = InferenceConfig2()


#Pre-trained COCO weights
cones_mod = os.path.join(models_dir, "mask_rcnn_orange_cones_0004.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(cones_mod):
    utils.download_trained_weights(cones_mod)

#Define model
cone_model = modellib.MaskRCNN(mode="inference", config=inference_config2, model_dir=models_dir)

#load weights
cone_model.load_weights(cones_mod, by_name=True)

# Cones class names
cone_class_names = ['BG', 'cone']


#---------------------------------------------------------------------------------------------#
# Process Video
#---------------------------------------------------------------------------------------------#


#---------------------------------------------#
#Load and get metadata

vid = imageio.get_reader("../input_videos/"+fname,  'ffmpeg')

#frames per second
if 'manual_fps' in locals():
    vid_fps = 240
else:
    vid_fps = vid.get_meta_data()['fps']


#total frames
if vid.get_meta_data()['nframes']==float('inf'):
    print("failed to pull nFrames from meta data...counting manually")
    temp = 0
    for num, image in enumerate(vid.iter_data()):
        temp += 1
    vid_nframe = temp
else:
    vid_nframe = vid.get_meta_data()['nframes']


#---------------------------------------------#
# Find cone locations

#Create single image with average RGB values of each pixel from sample of frames in the video
#max_sample = 30
#nsample = min(max_sample,vid_nframe)

nsample = vid_nframe #for now be safe and just use entire video
sample_ind = np.random.choice(vid_nframe, nsample,replace=False)

#blank array
average_image = (vid.get_data(0)*0).astype('float64')

#loop over sample and compute "average image"
for ind in sample_ind:
    average_image += vid.get_data(ind)/nsample



average_image = average_image.astype('uint8')

#Apply Mask R-CNN model to average image
cone_results = cone_model.detect([average_image], verbose=1)
cone_r = cone_results[0]

#We need 4 cones. Sometimes there are multiple cones identified on the same cone
cones_mask = cone_r['masks']

#get centroids of each cone mask
if cones_mask.shape[2]==4: #if have 4, we are good to go
    print("Correct number of cones identified...")
    cone_centroids = []
    for i in range(cones_mask.shape[2]):
        y_centroid, x_centroid = np.argwhere(cones_mask[:,:,i]==True).mean(axis=0)    
        cone_centroids.append([x_centroid,y_centroid])
elif cones_mask.shape[2]<4:  #no fix for if we identify <4 cones yet
    raise ValueError('Less than 4 cones were detected.');
else:
    #add method for K-means clustering (k=4) to suggest the 4 cone placements if 
    print('More than 4 cones were detected...Applying K-means clustering');
    initial_cone_centroids = []
    for i in range(cones_mask.shape[2]):
        y_centroid, x_centroid = np.argwhere(cones_mask[:,:,i]==True).mean(axis=0)    
        initial_cone_centroids.append([x_centroid,y_centroid])
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(initial_cone_centroids)
    cone_centroids = kmeans.cluster_centers_


#Define start/end gates
left_start_finish = np.sort([msk[0] for msk in cone_centroids])[0:2].mean() 
right_start_finish = np.sort([msk[0] for msk in cone_centroids])[2:4].mean()   

#---------------------------------------------#
# define sprinter in each frame


#function to define location of person
def findSprinter(img):
    
    # apply model
    results = person_model.detect([img], verbose=1)
    r = results[0]
    mask = r['masks']
    mask = mask.astype(int)
    
    #raise error
    if mask.shape[2]==0:
        #nobody in frame
        x_centroid = None
    elif mask.shape[2]>1:
        #more than one person detected occasionally 
        #similarly to the cones, they masks are usually right next to each other
        #in this case taking the centroid of both masks works fine
        #What is not okay is if the video has many people in the background
        print("...More than one person detected...Assuming close together and take the centroid of both")
        image_mask = img.copy()
        image_mask[:,:,0] = image_mask[:,:,0] * mask[:,:,0]
        
        image_mask = img.copy()
        for j in range(image_mask.shape[2]):
            image_mask[:,:,j] = image_mask[:,:,j] * mask[:,:,0]
        
        y_centroid, x_centroid = np.argwhere(image_mask[:,:,0]==1).mean(axis=0)  
        
    else:
        #only one person detected...proceed
        image_mask = img.copy()
        image_mask[:,:,0] = image_mask[:,:,0] * mask[:,:,0]
        
        image_mask = img.copy()
        for j in range(image_mask.shape[2]):
            image_mask[:,:,j] = image_mask[:,:,j] * mask[:,:,0]
        
        y_centroid, x_centroid = np.argwhere(image_mask[:,:,0]==1).mean(axis=0)  
        
    if x_centroid == None:
        return None, None, None;
    else:
        return y_centroid, x_centroid, image_mask;

#set some variables to fill in/update    
timing_zone_ind = [0]*vid_nframe
sec_count = 0

#loop over each frame to determine if person is in the timing zone
for f_num in range(vid_nframe):
    print("Processing frame "+ str(f_num) + " out of "+str(vid_nframe)+"..." )
    
    #load video frame
    ex_img = vid.get_data(f_num)
    
    #get sprinter location
    ex_y , ex_x, ex_mask = findSprinter(ex_img)
    
    if ex_y == None: #no person detected in frame
        print("............no person in frame")
        pass;
    else: #person detected in frame ... 
        if ex_x > left_start_finish and ex_x < right_start_finish: #... and in timing zone
            timing_zone_ind[f_num] += 1
            sec_count += 1/vid_fps


#Frames counted for timing
nFrames_timed = np.sum(timing_zone_ind)

#convert frames to seconds based on fps
seconds_timend = nFrames_timed / vid_fps
print("Time = " + str(round(seconds_timend,2)) + " sec" )


#---------------------------------------------#
#QC images to output

#Important Frames to consider (frames before/after enter and exit the timing zone)
in_zone_ind = [ i for i, x in enumerate(timing_zone_ind) if x==1] 
important_frames = [np.min(in_zone_ind)-1,np.min(in_zone_ind),np.max(in_zone_ind),np.max(in_zone_ind)+1]

#save image of QC with time

#output filename with original video name and measured time in title
out_qc_fname = fname.replace('.mp4','') + "_time_" + str(round(seconds_timend,3)).replace(".","_") +"s.png"

#plot relevant frames
fig, ax = plt.subplots(2,2,figsize=(12,6))
fig.suptitle("Key Frames when Transitioning In/Out of Timing Zone", fontsize=16)
ax = ax.ravel()

for frm in range(len(important_frames)):
    print(important_frames[frm])
    img= vid.get_data(important_frames[frm])
    results = person_model.detect([img], verbose=1)
    r = results[0]
    mask = r['masks']
    mask = mask.astype(int)
    
    image_mask = img.copy()
    for j in range(image_mask.shape[2]):
        image_mask[:,:,j] = image_mask[:,:,j] * mask[:,:,0]
    
    y_centroid, x_centroid = np.argwhere(image_mask[:,:,0]==1).mean(axis=0)    
    
    ax[frm].imshow(img)
    ax[frm].imshow(image_mask,alpha=.75)
    ax[frm].scatter( [msk[0] for msk in cone_centroids], [msk[1] for msk in cone_centroids],marker=".",color="orange")
    ax[frm].vlines(left_start_finish,10,average_image.shape[0]-10,color="orange",linestyles='dashed')
    ax[frm].vlines(right_start_finish,10,average_image.shape[0]-10,color="orange",linestyles='dashed')
    if x_centroid > left_start_finish and x_centroid < right_start_finish:
        ax[frm].vlines(x_centroid,ymin=10,ymax=image_mask.shape[0]-10,color="lightgreen")
        ax[frm].plot(x_centroid, y_centroid, marker='o', markersize=5, color="lightgreen")
        ax[frm].text(img.shape[1]/10,img.shape[0]/10,"Frame: "+str(important_frames[frm])+" (IN)",color="lightgreen",fontsize="xx-large")
    else:
        ax[frm].vlines(x_centroid,ymin=10,ymax=image_mask.shape[0]-10,color="red")
        ax[frm].plot(x_centroid, y_centroid, marker='o', markersize=5, color="red")
        ax[frm].text(img.shape[1]/10,img.shape[0]/10,"Frame: "+str(important_frames[frm])+" (OUT)",color="red",fontsize="xx-large")

#save figure
fig.savefig("../output/"+out_qc_fname)


