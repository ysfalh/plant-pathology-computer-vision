
#%%
import os # by default current chdir is the one with the .py
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from random import randint, random
import numpy as np
import torch
from torchvision import datasets, transforms # to transforms to tensor etc. and create our own datasets
import pandas as pd 

init_size = (2048,1365) # w*h
default_size = (224,224) # the size we want at the end 


def random_transformation(img):
    """
    The probability here depends on weather we include the original images or not.
    If we do, the probabilty should be higher.
    """
    x = random()
    if(x<0.6):
        img = img.filter(ImageFilter.GaussianBlur())
    img = colorJitter(img, p=1.0)
    #img = centerCrop(img, 0.5)
    img = horizontalFlip(img,0.40)
    img = verticalFlip(img,0.40)
    img = randomRotation(img,0.40)
    #
    img = transforms.RandomResizedCrop(default_size, scale=(0.4, 1.0), ratio=(0.75, 1.33333), interpolation=2)(img)
    #img = reSize(img)
    return img

# each following function returns an image that will be added to a new pull of image
crop_size = (1024, 682)

def centerCrop(img,p):
    x = random()
    if(x<p):
        return transforms.CenterCrop(crop_size)(img)
    else :
        return img

def colorJitter(img,p): # I should maybe add one more or something like that
    x = random()
    if(x<p):
        return transforms.ColorJitter(brightness=0.3, contrast=0.25)(img) # random
    else :
        return img

def horizontalFlip(img,p):
    return transforms.RandomHorizontalFlip(p=p)(img)

def randomRotation(img,p):
    size = (1024,656) # for example
    x = random()
    if(x<p):
        return transforms.RandomRotation(degrees = 40)(img) # default degree = 45
    else :
        return img


def verticalFlip(img,p):
    return transforms.RandomVerticalFlip(p=p)(img)

def reSize(img):
    return transforms.Resize(default_size, interpolation=2)(img)


# this functions is used to create the lines for the csv ... [Train_1, 0, 0, 1, 0] for example.
def toList(y, name):
    l = []
    l.append(name)
    for k in range(1,5):
        l.append(y[k])
    return l


#%%

"""
The goal here is : 
    -  Create an augmented set of images using the first n_train = 1092 (if we are doing 60-40 split)
    -  Everything gets to be automated - no adding of weird stuff etc.
    - 
"""


n=1821 # number of images in the initial set
path = "images/"
image_root_name = "Train_"  # "Test_" 
extension = ".jpg"

y_train = pd.read_csv("train.csv").values # the ground truths values

# what changes for each dataset :
path_to_save = "images_8/Train_" # where we save the augmented images # "Test_"
                                 # the folder has to be created first

# number of images to train with
n_train = 1400 # 1092 is for a 60/40 split - 1400 is for 77/23 split 
new_csv = [["image_id", "healthy","multiple_diseases","rust","scab"]] # always like this


# how many new images per class do we want ?
size_class_2 = 0   # you have to add 1 if you want have the total factor (because we automatically add the original picture)
size_other_classes = 0 

count = 0 # can be adjusted - count is the number that follows path_to_save (Train_count / Test_count) 
          # useful if we are adding new images to an existing dataset (in this case, csv will have to be merged)
        
for k in range(n_train): 
    img = Image.open(path+image_root_name+str(k)+extension)

    if(y_train[k][2]==1):
        for j in range(size_class_2):
            path_img = path_to_save + str(count) + extension
            img_ = random_transformation(img)
            img_.save(path_img)
            new_csv.append(toList(y_train[k],image_root_name+str(count)))
            count +=1
        
        path_img = path_to_save + str(count) + extension
        new_csv.append(toList(y_train[k],image_root_name+str(count)))
        count +=1
        img = reSize(img)
        img.save(path_img)
        
    else:
        for j in range(size_other_classes):
            path_img = path_to_save + str(count) + extension
            img_ = random_transformation(img)
            img_.save(path_img)
            new_csv.append(toList(y_train[k],image_root_name+str(count)))
            count +=1
        
        path_img = path_to_save + str(count) + extension
        new_csv.append(toList(y_train[k],image_root_name+str(count)))
        count +=1
        img = reSize(img)
        img.save(path_img)

        

#%%
import csv
print(len(new_csv))
name_new_csv = "train_8.csv"
with open(name_new_csv,"w") as outfile:
    out = csv.writer(outfile,delimiter=',')
    for row in new_csv:
        out.writerow(row)
#%%

# To merge dataqet
new_csv =[["image_id", "healthy","multiple_diseases","rust","scab"]]
y_train1 = pd.read_csv("train_4.csv").values
y_train2 = pd.read_csv("train_8_temp.csv").values
for k in range(len(y_train1)):
    new_csv.append(y_train1[k])
for k in range(len(y_train2)):
    new_csv.append(y_train2[k])

with open("train_8.csv","w") as outfile:
    out = csv.writer(outfile,delimiter=',')
    for row in new_csv:
        out.writerow(row)
#%%
