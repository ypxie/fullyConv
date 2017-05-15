## Folder Organization
Almost in all training, testing code, there will be a "BaseDir" parameters used to indicate the parent folder for all data and model.       
In current code, set as "BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')". There will be "NatureData" and "NatureModel" in this folder. "NatureData" is used for training/validation/testing data. "NatureModel" used for saving all trained models.
If trained on other machine, just need to change the "BaseDir" and copy all "NatureData" and "NatureModel" would work.

Currently, we have three experiment designs. 1. Indivisual and generic comparison. 2. Transfer to other dataset. 3. Compare training using limited dataset with using more data, or other dataset for improvement.

Under "NatureData" folder, there are three folders. 1. "YuanpuData" is used for saving data for "generic and indivisual" experiment. Under this folder, there are "TestingData", "TrainingData" as well as "ValidationData". There is an "Experiments" folder. All quantitative testing results are saved in it. "evaluation" and "evaluation_validation" is the results for "generic and indivisual" experiment. "evaluation_other" is for transfer experiment. "evaluation_pingjun" is the results for limited dataset experiment. 2. "OtherData" is for transfer experiment. There are "TestingData" and "TrainingData" folder in it for storing testing data and training data, seperately. No validation dataset is used here. 3. "PingjunData" for limited data experiment. There are "TrainingData" and "ValidationData" here, "ValidationData" is used to select the best model. Under "TrainingData", there are currently randomly selected 5 diseases. Under each disases, multiple combination of diseases for limited data training comparison. As testing dataset is the same with "YuanpuData", as these 5 diseases are selected from "YuanpuData".

Under "NatureModel" folder, three folders stands for the trained models for three designed experiments. "YuanpuModel" is for indivisual and generic comparison; "OtherModel" is for transfer experiemnt; "PingjunModel" for limited data experiemnts. One thing to note is, in "OtherModel", as there is no validation set, so the model name is "weights.pth", for "YuanpuModel" and "PingjunModel", the model used is "best_weights.pth", which was chose according to the performance on validation dataset.

As for the training, testing as well as plotting explanation, refer to description in corresponding sub-folders.

## Note
* To enable parallel training, python3 should be used.
