import torch
import os
from data_loader import captcha_images_v2_dataset
class train_NN:
  def __init__(self,epochs,batch_size,directory_training_data,
               directory_test_data,directory_val_data,
               Neural_Network,criterion,learning_rate):

    self.epochs = epochs
    self.batch_size = batch_size
    self.directory_training_data = directory_training_data
    self.directory_test_data = directory_test_data
    self.directory_val_data  = directory_val_data
    self.data_generator_train = captcha_images_v2_dataset(self.directory_training_data) 
    self.data_generator_test = captcha_images_v2_dataset(self.directory_test_data) 
    self.data_generator_val = captcha_images_v2_dataset(self.directory_val_data)
    self.Neural_Network = Neural_Network
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.optimizer = torch.optim.Adam(self.Neural_Network.parameters(), lr=self.learning_rate)

  def train(self):
    loss_total = 0
    for i in range(0,self.epochs):
      for steps in range(0,int(len(os.listdir(self.directory_training_data))/self.batch_size)):
        x_train , y_train = self.data_generator_train.__getitem__(batch_size=self.batch_size)
        x_train = torch.Tensor(x_train)
        x_train = x_train.view(x_train.shape[0], 3, x_train.shape[1], x_train.shape[2])
        y_train = torch.Tensor(y_train)
        y_pred  = self.Neural_Network(x_train)
        y_pred  = y_pred.permute(1,0,2)
        input_lengths = torch.IntTensor(self.batch_size).fill_(y_pred.size()[0])
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        self.optimizer.step()
        loss_total = loss_total + loss
      loss_total = loss_total/int(len(os.listdir(self.directory_training_data)))
      print('The training loss for epoch '+ str(i) + ' is '+ str(loss_total) )
      print('The validation loss for epoch '+str(i)+' is '+str(self.validation()))
    print('The testing loss model was '+str(self.test()))

  def test(self):
    loss_total = 0
    for steps in range(0,int(len(os.listdir(self.directory_test_data))/self.batch_size)):
        x_train , y_train = self.data_generator_test.__getitem__(batch_size=self.batch_size)
        x_train = torch.Tensor(x_train)
        x_train = x_train.view(x_train.shape[0], 3, x_train.shape[1], x_train.shape[2])
        y_train = torch.Tensor(y_train)
        y_pred  = self.Neural_Network(x_train)
        y_pred  = y_pred.permute(1,0,2)
        input_lengths = torch.IntTensor(self.batch_size).fill_(y_pred.size()[0])
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
        loss_total = loss_total + loss
    loss_total = loss_total/int(len(os.listdir(self.directory_test_data)))
    return loss_total

  def validation(self):
    loss_total = 0
    for steps in range(0,int(len(os.listdir(self.directory_val_data))/self.batch_size)):
        x_train , y_train = self.data_generator_val.__getitem__(batch_size=self.batch_size)
        x_train = torch.Tensor(x_train)
        x_train = x_train.view(x_train.shape[0], 3, x_train.shape[1], x_train.shape[2])
        y_train = torch.Tensor(y_train)
        y_pred  = self.Neural_Network(x_train)
        y_pred  = y_pred.permute(1,0,2)
        input_lengths = torch.IntTensor(self.batch_size).fill_(y_pred.size()[0])
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
        loss_total = loss_total + loss
    loss_total = loss_total/int(len(os.listdir(self.directory_val_data)))
    return loss_total
