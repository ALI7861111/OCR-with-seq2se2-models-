import optuna
from torch import nn
from src.seq2seq_NN import NeuralNetwork
from src.model_trainer import train_NN




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
        factor = trial.suggest_categorical("n_growth_factor_{}".format(layer), [1,2,3])
        growth_array.append(int(factor))

    number_layer_memory_unit = trial.suggest_categorical('number layer memory unit', [2])
    hidden_size_memory_unit_1 = trial.suggest_categorical('hidden size memory unit 1', [128])
    hidden_size_memory_unit_2 = trial.suggest_categorical('hidden size memory unit 2', [64])
    linear_layer_units_1 =  trial.suggest_categorical('linear layer units 1', [512])
    linear_layer_units_2 =  trial.suggest_categorical('linear layer units 2', [256])
    learning_rate =  trial.suggest_float('learning rate', 0,0.000000001)
    #batch_size    =  trial.suggest_int('batch_size', 16)
    batch_size    = trial.suggest_categorical('linear layer units 2', [16])

    print()

    try :
        CNN = NeuralNetwork(number_layers=int(nb_conv_layers),input_channels=3,batch_size = batch_size,
                            growth_factor=growth_array ,num_layers_memory_unit=int(number_layer_memory_unit),
                            input_size=(1,3,50,200), hidden_size_memory_unit_1=int(hidden_size_memory_unit_1),
                            hidden_size_memory_unit_2=int(hidden_size_memory_unit_2),linear_layer_units_1 = int(linear_layer_units_1),
                            linear_layer_units_2 = int(linear_layer_units_2))

        criterion = nn.CTCLoss(blank=4, reduction='mean', zero_infinity=True)

        trainer = train_NN(epochs=2,batch_size=batch_size,directory_training_data='/content/captcha_images_v2',
                      directory_test_data='/content/test',directory_val_data='/content/valid',
                      Neural_Network =CNN,criterion = criterion,learning_rate = learning_rate)
        loss = trainer.train()
        return loss

    except:

        return 0

    


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}