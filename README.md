# OCR-with-seq2se2-models
This repository contains Seq2Seq OCR development. The purpose of the repository is to compare and develop best model for OCR using Surrogate function. The features that are optimized to produce best OCR model are as follows. These features are Hyper-parameter and optimized based on surrogate function.

1.  Architectures 
2.  Learning rates
3.  Number of convolution layers
4.  Number of nodes in linear layers
5.  Number of convolution layers 
6.  Number of nodes in linear layers
7.  Growth relationship in convolution layers
8.  Number of layers in memory units
9.  Hidden size in memory units
10. Batch size

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

source = '/content/captcha_images_v2'
try:
  os.makedirs('/content/test')
except:
  print('Error Creating Directory')
dest = '/content/test'
files = os.listdir(source)
# Approximately 10 percent of the dataset
no_of_files = 90
try:
  for file_name in random.sample(files, no_of_files):
      shutil.move(os.path.join(source, file_name), dest)
except:
  print('Duplicate files exsist')
# Making validation dataset from all the data


source = '/content/captcha_images_v2'
try:
  os.makedirs('/content/valid')
except:
  print('Error Creating Directory')
dest = '/content/valid'
files = os.listdir(source)
# Approximately 10 percent of the dataset
no_of_files = 90
try:
  for file_name in random.sample(files, no_of_files):
      shutil.move(os.path.join(source, file_name), dest)
except:
  print('Duplicate files exsist')


```