# Datasets
Datasets can be added fairly easily by creating a compliant numpy label file and adding just a few lines of code. See the imagenet_video folder for an example. Given a path to the data, [imagenet_video/make_label_files.py](imagenet_video/make_label_files.py) creates the label files for the train and test set. To add new datasets, a similar file will be necessary. The other files that need to be modified are [get_datasets.py](../get_datasets.py) follow the example in the file, [batch_cache.py](../batch_cache.py) add a line like 168
```python
self.add_dataset('your_dataset_name_here')
```
and [unrolled_solver.py](../unrolled_solver.py) add a line like 158
```python
add_dataset('your_dataset_name_here')
```
These datasets should be listed in the same order in both files. 

Rather than putting your data in the repository, I recommend using simlinks (ln -s) or putting a base data path in the [constants.py](../../constants.py) file.


## Label file.
For each dataset, for each mode that you have (train/val/test) there should be a labels.npy file as well as a file listing all the image paths called image_names.txt. All labels.npy files should be of the format:

|0       |1       |2       |3       |4       |5       |6       |...     |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|xmin    |ymin    |xmax    |ymax    |video_id|track_id|im_id   |extra   |

The rows should be ordered such that tracks are sequential (rather than having all tracks in a single image sequentially ordered). If desired, the rows can be easily reordered using np.lexsort (see [imagenet_video/make_label_files.py](imagenet_video/make_label_files.py) for an example).

## Sources
The official ImageNet pages seem to be rapidly leaving the internet. Unfortunately, I cannot rehost them due to legal restrictions. 
If you are lucky, these links will still work when you try them.
- ImageNet Video Dataset: http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid
- ImageNet Object Detection (academic torrents) http://academictorrents.com/collection/imagenet-lsvrc-2015