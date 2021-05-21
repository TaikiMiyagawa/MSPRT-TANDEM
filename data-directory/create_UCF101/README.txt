#####################################################
# How to convert avi video files to TFRecords files #
#####################################################
Requirement: Pytorch 1.6.0 and microsoftvision 1.0.5

### 1. Preliminaries ###

1-1. Define DATADIR first.

1-2. Download 
"The UCF101 data set" 
and 
"The Train/Test Splits for Action Recognition on UCF101 data set" 
from https://www.crcv.ucf.edu/data/UCF101.php .

1-3. Extract UCF101. The directory tree must be like
DATADIR/UCF101/videos/{class directories}/{avi videos} 
and
DATADIR/UCF101/labels/ucfTrainTestlist/{classInd.txt etc.}.


### 2. Main Part ###

2-1. avi2png.sh: Convert avi files to png files. DATADIR/UCF101png/ will be created.
2-2. stat_org.ipynb: Optional. Check the statistics of the original dataset with stat_org.ipynb.
2-3. clipping.ipynb: Clip videos with a fixed length, which is referred to as "duration" in our codes. DATADIR/UCF101clip50 and DATADIR/UCF101clip150 will be created.
2-4. stat_clipped.ipynb: Optional. Check the statictics of the clipped dataset.
2-5. splitting_txt.ipynb: Define train/val/test splitting. DATADIR/UCF101/labels/ucfTrainValidTestlist/ will be made.
2-6. splitting_img.ipynb: Split videos following the train/val/test splitting. DATADIR/UCF101clip50tvt and DATADIR/UCF101clip150tvt will be created.
2-7. featextraction.py: Extract 2048-dim bottleneck features with microsoftvision.resnet50 and save them as npy files. The model tar file is stored in your current directory (MicrosoftVision.ResNet50.tar). DATADIR/UCF101-[duration]-[240-320 or 256-256] will be created.
2-8. Npy2TFR.ipynb: Convert npy files to a TFRecord file. DATADIR/UCF101TFR-[duration]-[240-320 or 256-256]/ will be made.