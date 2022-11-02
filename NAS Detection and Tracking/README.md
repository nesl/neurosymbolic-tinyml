

# Setup:

First, run pip install for the requirements.txt
```
pip install -r requirements.txt
```
Then, additionally install pycocotools and cython_bbox
```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

You will also need to install the latest DCNv2, which is part of the neural network object detection:
- First, clone https://github.com/jinfagang/DCNv2_latest
- Move it into updated_trackers/CenterTrack/src/lib/model/networks/DCNv2_latest
- Rename the folder as DCNv2
- Follow their install instructions

Then, download and move the MOT17 dataset, which we use for training and validation of pedestrian tracking:
- Download dataset from https://motchallenge.net/data/MOT17/
- Extract and rename folder as mot17
- Move folder to updated_tracks/CenterTrack/data/mot17

# Running NAS

Running Neural Architecture Search over different neural networks for object detection and tracking is quite easy - open a jupyter notebook:
```
jupyter notebook
```
And then open NAS_filterneural.ipynb to start playing around!


# Running Regular Training/Testing without NAS:


## Running Training:

By visiting updated_trackers/CenterTrack/src/lib/opts.py, you will see several command line arguments which can be used to alter the model structure.  Our neural model is a ResNet + DCN, and in our NAS we alter the number of convolutional layers, kernel sizes, number of output channels, and whether or not to use ReLU activations.  These are described as "actual_num_stacks", "kernel_size", "head_conv_value", and "activations" under opts.py.  You may also set these in addition to the following command line arguments.  

```
python main.py tracking --exp_id mot17_half_exp1 --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1 --num_epochs 70 --batch_size 16
```


## Running Testing:
```
python test.py tracking --exp_id mot17_half_exp1 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume
```


# Parameters for tracking:
pre_thresh:  [0, 1.0] - this determines whether the confidence of a detected unassociated item to be initialized as a new track (higher means harder to create the new track).  Default is 0.3

track_thresh: [0, 1.0] - this determines the confidence whether a detected item is even considered
for tracking (higher means harder to add an item to be tracked).  Default is 0.3.  This is different from pre_thresh in that usually pre_thresh >= track_thresh.  This means we can associate a detected item with a known track, but it may take higher confidence to create a new track.

These are parameters which can be added to your command line call (e.g. "python3 ... --new_thresh 0.4")



# References:

We obtained the CenterTrack source code from https://github.com/xingyizhou/CenterTrack.git
