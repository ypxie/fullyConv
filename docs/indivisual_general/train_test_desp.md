## Training
The training code is "yuanpu_train/multhread.py"
In this training, just set all 25 different diseases for indivisual models and "All" for the generic model. Then the generic and indivisual model would be trained. Just need to set "BaseDir". The trained models would be saved in "NatureModel/YuanpuModel"


## Testing
The testing code is "yuanpu_testing/nature_testing.py" and "yuanpu_testing/validation_testing.py", "nature_testing.py" is used for testing data. "validation_testing.py" is applied on validation data.
For both scripts, there is one parameter "parser.add_argument('--indvidual')", set this value as "True", it evaluates on indivisual disease model, if set as "False", evaluate on generical model. Need to run 4 times in total for generating the resutls. (testing, validation) * (indivisual, all). There will be two kinds of results, first kind is 'f1-score', 'precision' like quantitative results, there results are saved in "NatureData/YuanpuData/Experiments/evaluation" for testing, "NatureData/YuanpuData/Experiments/evaluation_validation" for evaluation_validation. Another kind of results is the detected points under different kind of threshold, these results are saved in its accompanied testing or validation folder, which can be used for drawing the detection results images.
