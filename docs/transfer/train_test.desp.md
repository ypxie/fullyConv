## Training
Set "BaseDir" for new training. Here we need to train two models (in total there are three, as the generic model is already trained), one is on indivisual data, the other is fine-tune on generic model. The code  is "yuanpu_train/train_other.py", by settting  "modelname" to "multicontex_ind" or "multicontex" means for "indivisual" and "fine-tune" model.


## Testing
The testing code is better it refer to "pingjun_test/otherdata_testing", though "yuanpu_testing" also has one.
As there would be three models to compare, there will be some combination for the parameters setting.
To test the generic model without fine-tune, just set "parser.add_argument('--indvidual')" to be "False", indivisual model as well as fine-tuned generic model, first set "parser.add_argument('--indvidual')" to be "True". Then for "indivusal model", use line 127/128 "multicontex_ind" model, for fine-tuned model, use 131/132 "multicontex" model. The quantitative reuslts would be saved in "NatureData/YuanpuData/Experiments/evaluation_other", accompanied cell centers would be saved in the same folder as the testing image data.

 
