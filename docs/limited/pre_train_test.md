## Data preprocessing
Need to get combination of disease. The code is in "preprocessing/limited_comparison_partition.py".
Five diseases are randomly selected. For each disease, first randomly select 5 images from training dataset as "Base". Then randomly select another disease from the rest 24 disease as "extra", randomly select 5 from extra combined with "Base" to form "Base+1Extra5", randomly select another 10 to combine with "Base+Extra5" to form "Base+1Extra15", apart from this two diseases, randomly select another two diseases, select 5 images randomly from these two diseases to compose "Base+3Extra5". Do this process again, to get another "Base+1Extra5", "Base+1Extra15" and "base+3Extra5", as there will be trained "indivisual" and "generic" model trained already in indivisal and comparsion experiment, so there will be 9 models there for each disease.

## Training
For each diseases, need to train separately. The parameters that need be set is "DiseaseName" in (line 13), "BaseDir" in (line 14), the training_pool (line 23) need to be set for which kind of combination you would like to train. The process_size and device_pool in line 28/29 is for parallel training. Trained model would be saved in "NatureModel/PingjunModel".

## Testing
The parameters need to set here is only "testing_folders" in line 128, add all these combinations you want to run the test code. The results would be saved in
"NatureData/YuanpuData/Experiments/evaluation_pinjun" as well as "NatureData/YuanpuData/TestingData".
