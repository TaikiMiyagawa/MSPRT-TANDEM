# MIT License

# Copyright (c) 2021 Taiki Miyagawa and Akinori F. Ebihara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ==============================================================================


# References (Japanese sites):
# https://qiita.com/livlea/items/a94df4667c0eb37d859f
# https://hacknote.jp/archives/20987/
# https://qiita.com/koara-local/items/04d3efd1031ea62d8db5

# Define DATADIR, 
# and then download and extract UCF101 under DATADIR.
# Directory tree must be like
# DATADIR/UCF101/videos/{class directories}/{avi videos} 
# DATADIR/UCF101/labels/ucfTrainTestlist/{classInd.txt etc.}

# DATADIR/UCF101png/... will be created.

# Create class directories.
echo "Let's create class directories!"
for f in DATADIR/UCF101/videos/*
do
    echo "Source class directory: $f"
    classname="${f#*UCF101/videos/}"
    echo "Class name: $classname"
    classdir_trg="DATADIR/UCF101png/$classname"
    echo "Target class directory: $classdir_trg"
    if [ ! -d $classdir_trg ]; then
        echo $classdir_trg does not exist.
        mkdir $classdir_trg
        echo "Created $classdir_trg."
    else
        echo "$classdir_trg already exists.\nSkipped."
    fi
    echo 
done
echo "Done. Next, let's convert avi to png!!"

# Create video directories.
# And ffmpeg .avi to .png.
for f in DATADIR/UCF101/videos/*
do
    # Extract class name, target class-directory name, and source class-directory
    classname="${f#*UCF101/videos/}"
    echo "Class name: $classname"

    classdir_trg="DATADIR/UCF101png/$classname"
    classdir_src="DATADIR/UCF101/videos/$classname"
    echo "Target class-directory: $classdir_trg"
    echo "Source class-directory: $classdir_src"

    # Loop for all videos in the source class-directory
    for ff in DATADIR/UCF101/videos/$classname/*.avi
    do
        # Extract video name and target video-directory.
        _tmp=${ff%.avi*}
        videoname=${_tmp#*$classname/}
        echo "Video name: $videoname"
        videodir=DATADIR/UCF101png/$classname/$videoname

        # If there does not exist the target video-directory, then create it.
        if [ ! -d $videodir ]; then
            mkdir $videodir
            echo "Created $videodir."
        else
            echo "Target video-directory ($videodir) already exists."
        fi

        # Conversion from avi to png
        # FPS is kept.
        # It takes a few tens of minutes.
        ffmpeg -i $ff -vcodec png $videodir/%04d.png

        echo "$ff done."
    done
    echo 
done
echo "Done."