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
9.  Optimizer

## Seq2Seq models 

Sequence to Sequence (often abbreviated to seq2seq) models is a special class of Recurrent Neural Network architectures that we typically use (but not restricted) to solve complex Language problems like Machine Translation, Question Answering, creating Chatbots, Text Summarization, etc.

Seq2Seq model are also used for development of the OCR development with CTC-LOSS. 

![Seq2seq for translation](https://miro.medium.com/max/1400/1*_rSHLjFShknAu3jt3rbcNQ.png)

## CTC-LOSS

A Connectionist Temporal Classification Loss, or CTC Loss, is designed for tasks where we need alignment between sequences, but where that alignment is difficult - e.g. aligning each character to its location in an audio file. It calculates a loss between a continuous (unsegmented) time series and a target sequence. It does this by summing over the probability of possible alignments of input to target, producing a loss value which is differentiable with respect to each input node. The alignment of input to target is assumed to be “many-to-one”, which limits the length of the target sequence such that it must be  the input length.

![CTC LOSS FUNCTION OUTPUT](https://miro.medium.com/max/1200/1*1_5KnLvaTkGUFoyat2jHcQ.png)


## Surrogate Functions

A surrogate model is an engineering method used when an outcome of interest cannot be easily directly measured. 
One way of alleviating this burden is by constructing approximation models, known as surrogate models. 

If we have an search space {s1,s2,s3....s200}

The agent/model takes only 4 varaibles from search space 
i.e. agent = MODEL(SX1,SX2,SX3,SX4)

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

The datset follows the rule. Every image with the captions inside it has its name with caption.png

If and image has A12DF written in it.It should be saved with the nemae A12DF.png

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
no_of_files = NUMBER OF FILES FOR TEST DATA
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
no_of_files = NUMBER OF FILES FOR VALIDATION DATA
try:
  for file_name in random.sample(files, no_of_files):
      shutil.move(os.path.join(source, file_name), dest)
except:
  print('Duplicate files exsist')


```


## Saving the best Hyper-parameter Trained Model

After getting the optimized parameters from the surrogate function saved into the csv.
Train your model consisting of the best Hyperparameters and save the model. This can be done by following code.


``` python

from torch import nn
import torch
from src.seq2seq_NN import NeuralNetwork
from src.model_trainer import train_NN
import argparse



parser = argparse.ArgumentParser(description="OCR Development")


parser.add_argument("--test_data_path", type=str, default='/content/test',
                    help='Testing data for the OCR')
parser.add_argument('--valid_data_path', type =str, default='/content/valid',
                    help='Validation data for the OCR')
parser.add_argument('--train_data_path', type =str, default='/content/train',
                    help='Training data for the OCR')
parser.add_argument('--epochs', type =int, default=2,
                    help='The Epochs for training Each Neural Network')
args = parser.parse_args()



test_path = args.test_data_path
valid_path =args.valid_data_path
train_path = args.train_data_path
epochs = args.epochs


trainer = train_NN(epochs=epochs,directory_training_data=train_path,
                  directory_test_data=test_path,directory_val_data=valid_path)



CNN = NeuralNetwork(number_layers=int(BEST OPTIMAL PARAMETER),input_channels=3,
                    batch_size = BEST BATCH SIZE,
                    growth_factor=BEST OPTIMAL PARAMETER ,num_layers_memory_unit=int(BEST OPTIMAL PARAMETER),
                    input_size=(1,3,50,200), hidden_size_memory_unit_1=int(BEST OPTIMAL PARAMETER),
                    hidden_size_memory_unit_2=int(BEST OPTIMAL PARAMETER),
                    linear_layer_units_1 = int(BEST OPTIMAL PARAMETER),
                    linear_layer_units_2 = int(BEST OPTIMAL PARAMETER), 
                    Unique_character_list = trainer.data_generator_train.Unique_character_list)

criterion = nn.CTCLoss(blank= 4, reduction='mean', zero_infinity=True)

loss = trainer.train(Neural_Network =CNN,criterion = criterion,learning_rate = BEST LEARNING RATE,batch_size=BEST BATCH SIZE, optimizer=BEST OPTIMIZER)

torch.save(CNN.state_dict(), PATH)

```




