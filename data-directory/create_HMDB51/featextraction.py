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

import os, shutil
from glob import glob

import numpy as np
import torch
import microsoftvision
from torchvision import transforms
from PIL import Image

def size_adjustment(imgs, shape):
    """
    Args:
        imgs: Numpy array with shape (data, width, height, channel)
            = (*, 240, 320, 3).
        shape: 256 or None.
            256: imgs_adj.shape = (*, 256, 256, 3)
            None: No modification of imgs.
    Returns:
        imgs_adj: Numpy array with shape (data, modified width, modified height, channel)
    """
    if shape is None:
        imgs_adj = imgs
        
    elif shape == 256:
        # Reshape from 240x320 to 256x256
        imgs_adj = np.delete(imgs, obj=[i for i in range(32)] + [i for i in range(287, 319)], axis=2)

        _tmp = imgs_adj.shape
        mask = np.zeros(shape=(_tmp[0], 8, _tmp[2], _tmp[3]), dtype=np.uint8)
        imgs_adj = np.concatenate([imgs_adj, mask], axis=1)
        imgs_adj = np.concatenate([mask, imgs_adj], axis=1)
    
    return imgs_adj

def normalize_images(imgs, device):
    """ Input = channel last!!
    Args: 
        imgs: Numpy array with shape (duraiton, width, height, channel)
        devide: A torch,device object.
    Return:
        tens: Torch tensor with shape (duration, channel, width, height).
            A normalized video with the magic numbers.
    """
    preprocess = transforms.Compose([
            transforms.ToTensor(), # convert input to channel first and normalize 0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    tens = []
    for v in imgs:
        tens.append(preprocess(v).unsqueeze(0))       
            # Unsqueeze only required when there's 1 image in images batch       

    tens = torch.cat(tens, dim=0).to(device)
    return tens


# User-defined
# ==========================================
gpu = 0 # which GPU to be used
batch_size = 1 
splitnum = 1 # Official split num. 1, 2, or 3.
duration = 79 # 79 or 200.
tvt = ["train", "valid", "test"][0] # [0], [1], [2]. Either of the three will be created..
target_imshape = None # None or 256, but 256 is not currently supported.
# ==========================================

trclippaths = sorted(glob("{}/HMDB51clip{}tvt/train0{}/*/*".format(DATADIR, duration, splitnum)))
vaclippaths = sorted(glob("{}/HMDB51clip{}tvt/valid0{}/*/*".format(DATADIR, duration, splitnum)))
teclippaths = sorted(glob("{}/HMDB51clip{}tvt/test0{}/*/*".format(DATADIR, duration, splitnum)))

# Define save directory
if target_imshape is None:
    savedir = "{}/HMDB51-{}-240-320".format(DATADIR, duration)
elif target_imshape == 256:
    savedir = "{}/HMDB51-{}-256-256".format(DATADIR, duration)
else:
    raise ValueError
    
assert not ".npy" in glob(savedir + "/*/*/*"), "Npy file exists under {}. Remove it.".format(savedir)
assert duration % batch_size == 0
assert batch_size <= duration

# Avoid overwriting npy files
assert not ".npy" in glob(savedir + "/*/*/*"), "Npy file exists under {}. Remove it.".format(savedir)
assert not ".npy" in glob(savedir + "/*/*"), "Npy file exists under {}. Remove it.".format(savedir)
assert not ".npy" in glob(savedir + "/*"), "Npy file exists under {}. Remove it.".format(savedir)

# Verbose
print("savedir =", savedir)
print("tvt =", tvt)

# Model instantiation
device = torch.device("cuda:{}".format(gpu))
model = microsoftvision.models.resnet50(pretrained=True, map_location=device)
model.to(device)
model.eval() 

# Train/val/test loop
if tvt == "train":
    clippaths = trclippaths
elif tvt == "valid":
    clippaths = vaclippaths
elif tvt == "test":
    clippaths = teclippaths
else:
    raise ValueError(tvt)
    
# All-clip loop
for _c, clippath in enumerate(clippaths):
    # Verbose 
    if (_c + 1) % 100 == 0:
        print("{} clip iter {}/{}: {}".format(tvt, _c + 1, len(clippaths), clippath))

    # Load images in a clip
    ################################
    imgpaths = sorted(glob(clippath.replace("[", "[[").replace("]", "[]]").replace("[[", "[[]") + "/*.png")) # images in a clip
    assert len(imgpaths) == duration, "Num of images in {} is {}".format(clippath, len(imgpaths))    
    imgs = np.array([np.array(Image.open(v)) for v in imgpaths]) 
        # (duration, width, height, channel)
        # e.g., (50, 240, 320, 3) for HMDB51clip79

    # Size change
    ###############################
    imgs_adj = size_adjustment(imgs, shape=target_imshape)
        # (duration, width', height', channel)

    # Extract features
    ###############################
    tens = normalize_images(imgs_adj, device)
        # (duration, channel, width', height')
    feats = []
    for i in range(int(duration / batch_size)):
        _in = tens[i * batch_size: (i + 1) * batch_size ]
        _out = model(_in) 
            # shape = (batch_size, 2048)
        feats.append(_out.cpu().detach().numpy())
    feats = np.concatenate(feats, axis=0)
        # (duration, 2048)
        
    # Define save paths
    ###############################
    oldpath = imgpaths[0]
        # 'DATADIR/HMDB51clip79/wave/prideandprejudice1_wave_f_nm_np1_ri_med_14_cc02/0079.png'
    oldpath2 = oldpath[: oldpath.rfind("/")]
        # 'DATADIR/HMDB51clip79/wave/prideandprejudice1_wave_f_nm_np1_ri_med_14_cc02'
    clipname = oldpath2[oldpath2.rfind("/") + 1:]
        # 'prideandprejudice1_wave_f_nm_np1_ri_med_14_cc02'
    oldpath3 = oldpath2[:oldpath2.rfind("/")]
        # 'DATADIR/HMDB51clip79/wave'
    classname = oldpath3[oldpath3.rfind("/") + 1:]
        # 'wave'

    saveclsdir = savedir + "/" + tvt + "0{}/".format(splitnum) + classname 
    savepath = saveclsdir + "/" + clipname + ".npy"
    if not os.path.exists(saveclsdir):
        os.makedirs(saveclsdir)

    np.save(savepath, feats)