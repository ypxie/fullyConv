clear; close all;
debugcount = 0;
copyRoot = fullfile('..','..','..','..');
%TestingRoot =  fullfile('/home/yuanpuxie/Dropbox/WorkStation/torch-fcn/Data/TestingData/CropData');
TestingRoot =  fullfile('/media/yuanpuxie/lab_drive/patches');

%TestingRoot =  fullfile('/home/yuanpuxie/Dropbox/WorkStation/torch-fcn/Data/TestingData/CropData');
%TestingRoot =  fullfile('/media/yuanpuxie/LinlinGao-BME360/');
TestingRoot = '/media/yuanpuxie/Seagate Backup Plus Drive/TCGA_data/CropImage/CropImage';
proj_root = fullfile('..');
addpath(fullfile('.', 'UglyCode'));

teststate = 'multicontex_weights';
dataset = 'Liver';

%test_tuple('ovary', ['.tif'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True),
%test_tuple('pancreas', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True),
%test_tuple('pleura', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True),
%test_tuple('prostate', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True)
                    
imgExt = {'.tif'};
radius = 5;

imgFolder = fullfile(TestingRoot , dataset);

ResultExt = ['_',teststate,'.mat'];
thisSeedDir = fullfile(imgFolder, teststate) ;  

[imgInfo,numImgs] = Read_Dir_Files(imgFolder,imgExt);


for img_idx = 1:1: numImgs
        imgname = imgInfo{img_idx}.name
        img = imread(imgInfo{img_idx}.fullname); 
        
        mat_path  = fullfile(imgFolder, [imgname, '_withcontour.mat']);
        ResultDictfile = fullfile(thisSeedDir, [imgname, ResultExt]);
        this_my_tmp  = load(ResultDictfile);
        
        distCount = 0;
        thisthresh = 0.25;

        this_seed_Entry = [teststate,'_s_', sprintf('%02d',1), '_t_',sprintf('%01.02f', thisthresh) ,'_r_', sprintf('%02.02f', radius)];
       
        this_seed_Entry(strfind(this_seed_Entry,'.')) ='_';
        
         this_my = this_my_tmp; %load(ResultDictfile);
         tmpseed = double(this_my.(this_seed_Entry));
         
         num_seeds = length(tmpseed);
         Contours= {};
         for idx= 1:1:num_seeds
              
              [fdX, fdY]=circlepoints(tmpseed(idx, 2), tmpseed(idx, 1), 8);
              Contours{idx} = [fdX';fdY'];
         end
         
         save(mat_path, 'Contours');
end
     