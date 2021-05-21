#####################################################
# How to convert avi video files to TFRecords files #
#####################################################
Requirement: Pytorch 1.6.0 and microsoftvision 1.0.5

### 1. Preliminaries ###

1-1. Define DATADIR first. E.g., DATADIR = "/data/t-miyagawa".

1-2. Download 
"HMDB51 – About 2GB for a total of 7,000 clips distributed in 51 action classes" 
and 
"three splits for the HMDB51" 
from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads .

1-3. Extract HMDB51. The directory tree must be like
DATADIR/HMDB51/videos/{class directories}/{avi videos} 
and
DATADIR/HMDB51/labels/{class_split.txt etc.}.


### 2. Main Part ###

2-1. avi2png.sh: Convert avi files to png files. DATADIR/HMDB51png/ will be created.
2-2. stat_org.ipynb: Optional. Check the statistics of the original dataset with stat_org.ipynb.
2-3. clipping.ipynb: Clip videos with a fixed length, which is referred to as "duration" in our codes. DATADIR/HMDB51clip79 and DATADIR/HMDB51clip200 will be created.
2-4. stat_clipped.ipynb: Optional. Check the statictics of the clipped dataset. 
2-5. splitting_txt.ipynb: Define train/val/test splitting. DATADIR/HMDB51/labelstvt/ will be created.
DATADIR/HMDB51clip79tvt and DATADIR/HMDB51clip1200tvt will be created.
2-6. splitting_img.ipynb: Split videos following the train/val/test splitting. DATADIR/HMDB51clip79tvt and DATADIR/HMDB51clip200tvt will be created.
2-7. featextraction.py: Extract 2048-dim bottleneck features with microsoftvision.resnet50 and save them as npy files. The model tar file is stored in your current directory (MicrosoftVision.ResNet50.tar). DATADIR/HMDB51-[duration]-240-320 will be created.
2-8. Npy2TFR.ipynb: Convert npy files to a TFRecord file. DATADIR/UCF101TFR-79-240-320/ will be made.