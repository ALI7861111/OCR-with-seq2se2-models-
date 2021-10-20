# OCR-with-seq2se2-models
This repository contains OCR development and benchmarking dataset. The purpose of the repository is to compare different approaches i.e. 3D Convolutions, Self Attentions and different architectures and  benchmarking response on the dataset  


## Dataset

You can use custom dataset for this repository but if custom dataset is not avaliable than you can download dataset for captcha images from 

https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip

or you can run the below command to automatically download and unzip the dataset. 

```bash

curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
unzip -qq captcha_images_v2.zip

```

Than you can create test train and validation data folders from the unzip dataset.

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