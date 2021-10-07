import cv2
import glob
import random
import os
class captcha_images_v2_dataset():
    """Captcha Images V2 Dataset"""
    def __init__(self, path_to_dataset):
        """
        The constructor takes the path to the dataset and initiates two list with corresponding images and labels. 
        """
        self.path_to_dataset = path_to_dataset

        self.get_unique_characters()

        self.images = [cv2.imread(file) for file in glob.glob(str(self.path_to_dataset)+"/*.png")]
        self.images_names = [self.get_number_encoding(file) for file in glob.glob(str(path_to_dataset)+"/*.png")]
        self.length = len(self.images)

    def get_unique_characters(self):
        Train_list = os.listdir(self.path_to_dataset)
        self.Unique_character_list = []
        for words in Train_list:
            words.replace('.png','')
            for character in words:
              if character not in self.Unique_character_list:
                self.Unique_character_list.append(character)
    
    
    def get_number_encoding(self,file_name):
        file_name.split('/')[-1].replace('.png','')
        ENCODING_ARRAY = []
        #for name in file:
        for character in file_name:
            if character in self.Unique_character_list:
              index = self.Unique_character_list.index(character)
              ENCODING_ARRAY.append(index)
        return ENCODING_ARRAY
    

    def __getitem__(self,batch_size=16):
        images_to_return = []
        labels_to_return = []
        for i in range(0,batch_size):            
          n = random.randint(0,self.length-1)
          images_to_return.append(self.images[n])
          labels_to_return.append(self.images_names[n])
        return (images_to_return,labels_to_return)