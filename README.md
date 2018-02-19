# DL Programing Assigment's README
1. Before running run.sh please change the paths --save_dir, --exp_dir, --mnist according to your machine.
2. Final trained model will be saved in "FinalTrainedModel.pickle" in --save_dir. The pickle file will have one list for all weights and one list for all biases. 
3. The training will stop after maximum no of epochs (--epochs hyperparameter) (by default 200) or after three contigues annealing.