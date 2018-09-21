# Re3
<img src="/demo/sample_1.gif" height="300"/>
<img src="/demo/sample_2.gif" height="300"/>


Re3 is a real-time recurrent regression tracker. It offers accuracy and robustness similar to other state-of-the-art trackers while operating at 150 FPS. For more details, contact xkcd@cs.washington.edu. This repository implements the training and testing procedure from https://arxiv.org/pdf/1705.06368.pdf. A sample of the tracker can be found here: https://youtu.be/RByCiOLlxug.

## Recent Updates:
I have changed the network design to be smaller than the one presented in the paper without loss in accuracy. If you want to use the smaller, faster network, you should do a git pull and download the new model linked to in the models section of the readme.

## Model:
The new smaller pretrained weights model is available for download at http://bit.ly/2L5deYF. Please download it, unzip it, and place it in your logs directory. For more instructions see the First Time Setup section.
If you would like to check the download, the md5 checksum is 10ec7fd551c30ea65a7deebe12bd532f.

## Requirements:
1. Python 2.7+ or 3.5+.
2. [Tensorflow](https://www.tensorflow.org/) and its requirements. I use the pip tensorflow-gpu==1.5.0
3. [NumPy](http://www.numpy.org/). The pip should work.
4. [OpenCV 2](http://opencv.org/opencv-2-4-8.html). The opencv-python pip should work.
5. [CUDA (Strongly Recommended)](https://developer.nvidia.com/cuda-downloads).
6. [cuDNN (Recommended)](https://developer.nvidia.com/cudnn).

## First Time Setup:
Here are sample steps for setup on Ubuntu. For another operating system, replace the apt-get command with however you normally install stuff.
```bash
git clone git@gitlab.cs.washington.edu:xkcd/re3-tensorflow.git
cd re3-tensorflow
sudo apt-get install python-virtualenv
virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Then go to http://bit.ly/2L5deYF and download the file. Finally:
```bash
mkdir logs
mv /path/to/downloads/checkpoints.tar.gz logs
cd logs
tar -zxvf checkpoints.tar.gz
```

### Enter the virtualenv in a later session to use the installed libraries.
```bash
source venv/bin/activate
```
### To exit the virtualenv
```bash
deactivate
```

## Folders and Files:
### Most important for using Re3 in a new project:
1. [tracker.py](tracker/re3_tracker.py)
2. [network.py](tracker/network.py)
3. [The provided model weights file.](https://goo.gl/NWGXGM)

### Most important for (re)training Re3 on new data:
1. [unrolled_solver.py](training/unrolled_solver.py)
2. [batch_cache.py](training/batch_cache.py)

### Most useful not related to Re3:
1. [tf_util.py](re3_utils/tensorflow_util/tf_util.py)
2. [tf_queue.py](re3_utils/tensorflow_util/tf_queue.py)
3. [drawing.py](re3_utils/util/drawing.py)

### More details than you wanted:
1. demo:
    * [image_demo.py](demo/image_demo.py) - A premade demo to show Re3 tracking an object.
    * [batch_demo.py](demo/batch_demo.py) - A premade demo to show Re3 tracking one or multiple objects.
    * [webcam_demo.py](demo/webcam_demo.py) - A demo that uses a webcam feed to track an object. Simply click and drag the mouse around the object to track. When you release the mouse, tracking will begin. To track a new object, simply draw a new box.
2. re3_utils:
    * simulater - The code that defines the simulation of object tracks.
    * tensorflow_util:
        * [CaffeLSTMCell.py](re3_utils/tensorflow_util/CaffeLSTMCell.py) - The custom cell ported from Caffe defining Re3's LSTM unit.
        * [tf_queue.py](re3_utils/tensorflow_util/tf_queue.py) - A nice wrapper for threaded data loading in Tensorflow. Feel free to use separately from Re3 as well.
        * [tf_util.py](re3_utils/tensorflow_util/tf_util.py) - A bunch of functions that should probably be built into Tensorflow, but mostly aren't. This includes things like grouped convolutions, automatic variable summarization, PReLU non-linearity, and a restore function that doesn't suck (modeled after Caffe's restore). Again, feel free to use this even if you don't use Re3.
    * util:
        * [bb_util.py](re3_utils/util/bb_util.py) - A few useful functions for bounding boxes in Numpy
        * [drawing.py](re3_utils/util/drawing.py) - A few custom drawing functions mainly used for showing outputs in one image.
        * [im_util.py](re3_utils/util/im_util.py) - A few functions for resizing images. get_cropped_input is used in many locations in the code as the function that crops, warps, and resizes images to be fed into the network. It is quite fast and robust.
        * [IOU.py](re3_utils/util/IOU.py) - A few functions for computing IOUs on Numpy arrays.
3. tracker:
    * [network.py](tracker/network.py) - The Re3 network definition in Tensorflow. This includes training and testing networks and operations.
    * [re3_tracker.py](tracker/re3_tracker.py) - A (roughly) stand-alone wrapper for the Re3 tracker for use at inference time. It can keep track of multiple tracks at once. Look in the demo folder for how to use it. It wraps all the Tensorflow ugliness, returning a nice, simple bounding box for every provided image. You should not instantiate a new tracker for each new object. Instead, use the unique_id given to track.
    * [re3_vot_tracker.py](tracker/re3_vot_tracker.py) - A simple interface to the VOT evaluation code that uses the re3_tracker.
4. training:
    * [batch_cache.py](training/batch_cache.py) - A local data server useful for reading in images from disc onto RAM. The images will be sent over a pipe to the training process.
    * [caffe_to_tf.py](training/caffe_to_tf.py)  - The script used to port Caffe weights to Tensorflow.
    * [read_gt.py](training/read_gt.py)  - Reads in data stored in a variety of different formats into a common(ish) format between all datasets.
    * [test_net.py](training/test_net.py)  - Runs through a provided sequence of images, tracking and computing IOUs on all ground truth frames. It can also make videos of the results.
    * [unrolled_solver.py](training/unrolled_solver.py)  - The big, bad training file. This unwieldy piece of code encapsulates much of the complexity involved in training Re3 in terms of data management, image preprocessing, and parallelization. It also contains lots of code for logging and debugging that I found immensely useful. Look through the command-line arguments and constants declared at the top for more knobs to tune during training. If you kill a training run with Ctrl-c, it will save the current weights (which I also found very useful) before it extits.
5. [constants.py](constants.py) - A place to put constants that are used in multiple files such as image size and log location.

## License and Citation

Re3 is released under the GPL V3.

Please cite Re3 in your publications if it helps your research:
```
@article{gordon2018re3,
  title={Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects},
  author={Gordon, Daniel and Farhadi, Ali and Fox, Dieter},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={2},
  pages={788--795},
  year={2018},
  publisher={IEEE}
}
```
