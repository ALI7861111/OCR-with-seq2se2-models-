import optuna
from torch import nn
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




def objective(trial):

    nb_conv_layers = trial.suggest_categorical('number of convolution layers', [1,2,3])
    growth_array   = []
    for layer in range(0,nb_conv_layers):
        factor = trial.suggest_categorical("n_growth_factor_{}".format(layer), [1,2])
        growth_array.append(int(factor))

    number_layer_memory_unit = trial.suggest_categorical('number layer memory unit', [1,2,3])
    #number_layer_memory_unit = 2
    hidden_size_memory_unit_1 = trial.suggest_categorical('hidden size memory unit 1', [64,128,256])
    #hidden_size_memory_unit_1 = 128
    hidden_size_memory_unit_2 = trial.suggest_categorical('hidden size memory unit 2', [64,128,256])
    #hidden_size_memory_unit_2 = 64
    linear_layer_units_1 =  trial.suggest_categorical('linear layer units 1', [128,256,512])
    #linear_layer_units_1 = 512
    linear_layer_units_2 =  trial.suggest_categorical('linear layer units 2', [128,256,512])
    #linear_layer_units_2 = 256
    learning_rate =  trial.suggest_float('learning rate', 0,0.000000001)
    batch_size    =  trial.suggest_categorical('batch_size', [16,32,64,128])
    #batch_size    = trial.suggest_categorical('linear layer units 2', [16])
    #batch_size = 16


    try :
        trainer = train_NN(epochs=epochs,batch_size=batch_size,directory_training_data=train_path,
                  directory_test_data=test_path,directory_val_data=valid_path)

        CNN = NeuralNetwork(number_layers=int(nb_conv_layers),input_channels=3,batch_size = batch_size,
                            growth_factor=growth_array ,num_layers_memory_unit=int(number_layer_memory_unit),
                            input_size=(1,3,50,200), hidden_size_memory_unit_1=int(hidden_size_memory_unit_1),
                            hidden_size_memory_unit_2=int(hidden_size_memory_unit_2),linear_layer_units_1 = int(linear_layer_units_1),
                            linear_layer_units_2 = int(linear_layer_units_2), Unique_character_list = trainer.data_generator_train.Unique_character_list)
        criterion = nn.CTCLoss(blank= 4, reduction='mean', zero_infinity=True)
        loss = trainer.train(Neural_Network =CNN,criterion = criterion,learning_rate = learning_rate)
        return loss

    except:
        print('Failed')
        # if the construction of the neural Networks fails the loss returned should be high as a indicator to surrogate function that parameters
        # were the worst paramteres.
        return 100

    


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=100)

df = study.trials_dataframe()
df.to_csv('Results.csv')


