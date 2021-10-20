import optuna
from torch import nn
from src.seq2seq_NN import NeuralNetwork
from src.model_trainer import train_NN
import argparse
import torch


parser = argparse.ArgumentParser(description="OCR Development")


parser.add_argument("--test_data_path", type=str, default='/content/test',
                    help='Testing data for the OCR')
parser.add_argument('--valid_data_path', type =str, default='/content/valid',
                    help='Validation data for the OCR')
parser.add_argument('--train_data_path', type =str, default='/content/train',
                    help='Training data for the OCR')

args = parser.parse_args()



test_path = args.test_data_path
valid_path =args.valid_data_path
train_path = args.train_data_path




trainer = train_NN(epochs=2,batch_size=16,directory_training_data=train_path,
                  directory_test_data=test_path,directory_val_data=valid_path)

def objective(trial):
    # CNN = NeuralNetwork(number_layers=3,input_channels=3,batch_size = 16,
    #                 growth_factor=growth_array,num_layers_memory_unit=2,
    #                 input_size=(1,3,50,200), hidden_size_memory_unit_1=128,
    #                 hidden_size_memory_unit_2=64,linear_layer_units_1 = 512,
    #                 linear_layer_units_2 = 256)
    
    #Sucess = False
    #while Sucess == False: 
    nb_conv_layers = trial.suggest_categorical('number of convolution layers', [1,2,3])
    growth_array   = []
    for layer in range(0,nb_conv_layers):
        factor = trial.suggest_categorical("n_growth_factor_{}".format(layer), [1,2])
        growth_array.append(int(factor))

    #number_layer_memory_unit = trial.suggest_categorical('number layer memory unit', [2])
    number_layer_memory_unit = 2
    #hidden_size_memory_unit_1 = trial.suggest_categorical('hidden size memory unit 1', [128])
    hidden_size_memory_unit_1 = 128
    #hidden_size_memory_unit_2 = trial.suggest_categorical('hidden size memory unit 2', [64])
    hidden_size_memory_unit_2 = 64
    #linear_layer_units_1 =  trial.suggest_categorical('linear layer units 1', [512])
    linear_layer_units_1 = 512
    #linear_layer_units_2 =  trial.suggest_categorical('linear layer units 2', [256])
    linear_layer_units_2 = 256
    learning_rate =  trial.suggest_float('learning rate', 0,0.000000001)
    #batch_size    =  trial.suggest_int('batch_size', 16)
    #batch_size    = trial.suggest_categorical('linear layer units 2', [16])
    batch_size = 16


    try :

        CNN = NeuralNetwork(number_layers=int(nb_conv_layers),input_channels=3,batch_size = batch_size,
                            growth_factor=growth_array ,num_layers_memory_unit=int(number_layer_memory_unit),
                            input_size=(1,3,50,200), hidden_size_memory_unit_1=int(hidden_size_memory_unit_1),
                            hidden_size_memory_unit_2=int(hidden_size_memory_unit_2),linear_layer_units_1 = int(linear_layer_units_1),
                            linear_layer_units_2 = int(linear_layer_units_2), Unique_character_list = trainer.data_generator_train.Unique_character_list)
        criterion = nn.CTCLoss(blank=4, reduction='mean', zero_infinity=True)
        loss = trainer.train(Neural_Network =CNN,criterion = criterion,learning_rate = learning_rate)
        return loss

    except:
        print('Failed')
        return 100

    


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=100)

#study.best_params  # E.g. {'x': 2.002108042}

# trainer = train_NN(epochs=10,batch_size=16,directory_training_data=train_path,
#                    directory_test_data=test_path,directory_val_data=valid_path)

# growth_array = [1,2,2]

# CNN = NeuralNetwork(number_layers=3,input_channels=3,batch_size = 16,
#                     growth_factor=growth_array,num_layers_memory_unit=2,
#                     input_size=(1,3,50,200), hidden_size_memory_unit_1=128,
#                     hidden_size_memory_unit_2=64,linear_layer_units_1 = 512,
#                     linear_layer_units_2 = 256, Unique_character_list = trainer.data_generator_train.Unique_character_list)

# criterion = nn.CTCLoss(blank=5, reduction='mean', zero_infinity=True)
# optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)




# trainer.train(Neural_Network =CNN,criterion = criterion,learning_rate = 0.0001)