# OCR-with-seq2se2-models
This repository contains Seq2Seq OCR development. The purpose of the repository is to compare and develop best model for OCR using Surrogate function. The features that are optimized to produce best OCR model are as follows. These features are Hyper-parameter and optimized based on surrogate function.

1.  Architectures 
2.  Learning rates
3.  Number of convolution layers 
4.  Number of nodes in linear layers
5.  Growth relationship in convolution layers
6.  Number of layers in memory units
7.  Hidden size in memory units
8.  Batch size

## Surrogate Functions

A surrogate model is an engineering method used when an outcome of interest cannot be easily directly measured. 
One way of alleviating this burden is by constructing approximation models, known as surrogate models. 

If we have an search space {s1,s2,s3....s200}

The agent/model takes only 4 varaibles from search space 
A = MODEL (SX1,SX2,SX3,SX4)

All the 200! ( permuation ) combinations cannot be applied. So the surrogate function will take some trails 
and approximation to get the minimum Error E.

#### Sudo Code For surrogate function
```
Search_space = {s1,s2,s3....s200}
Total_trails = N
loss = None
while trails =< total_trails { 

Sx1,Sx2,sx3,sx4 = surrogate_function.suggest(Search_space,loss)

MODEL = MODEL(sx1,sx2,sx3,sx4)
loss  = MODEL.train()  
  
}
```

The model will try to get the best parameters that minimize the loss. This technique is used to exploit the best hyper-parameter from a given search space of hyper-parameters. 


## Dataset

You can use custom dataset for this repository but if custom dataset is not avaliable than you can download dataset for captcha images from 

https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip

or you can run the below command to automatically download and unzip the dataset. 

```bash

curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
unzip -qq captcha_images_v2.zip

```

Than you can create test train and validation data folders from the unzip dataset.

Enter your desired Source, train, test and validation directories in the code below

``` python

import os
import random
import shutil

# Getting test dataset from all the data
# Path given according to jupyter files you can add your custom path

# The source directory is the images for training images 

source = PATH TO THE SOURCE IMAGES
try:
  os.makedirs(PATH TO THE TEST IMAGES)
except:
  print('Error Creating Directory')
dest = PATH TO THE TEST IMAGES
files = os.listdir(source)
# Approximately 10 percent of the dataset
no_of_files = 90
try:
  for file_name in random.sample(files, no_of_files):
      shutil.move(os.path.join(source, file_name), dest)
except:
  print('Duplicate files exsist')
# Making validation dataset from all the data


source = PATH TO THE SOURCE IMAGES
try:
  os.makedirs(PATH TO THE VALIDATION IMAGES)
except:
  print('Error Creating Directory')
dest = PATH TO THE VALIDATION IMAGES
files = os.listdir(source)
# Approximately 10 percent of the dataset
no_of_files = 90
try:
  for file_name in random.sample(files, no_of_files):
      shutil.move(os.path.join(source, file_name), dest)
except:
  print('Duplicate files exsist')


```