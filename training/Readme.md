# Training
## Commands
Typically you will need to run two processes to train, the batch_cache and the unrolled_solver. In one window, run
```
python batch_cache.py -n 2
```
Which says to run batch_cache with length 2 unrolls.
In another window run
```
python unrolled_solver.py -rtc -n 2 -b 64
```
Which says to run the solver with length 2 unrolls, a batch size of 64, to restore from a checkpoint if it exists, to show timing information, and to delete old checkpoints after new ones are saved (this does not use Tensorflow's method and will delete ALL older checkpoints in the checkpoint folder).
To view all the options for both, simply run with the `-h` flag.
For debugging, it is often useful to see the network's output on the input images. For this use the `-o` flag.
To change GPU, use `-v GPU_ID`. For increased speed, you can set 2 GPU_IDs and the self-training network will run on a separate GPU but will share parameters with the main network.
When training with 16/32 unrolls, it is probably a good idea to run the val process as well. This runs the current tracker against a validation set using the test_net.py script. To do this, add the `--run_val` flag. A common command towards the end of training would be
```
python unrolled_solver.py -rtc -n 32 -b 8 -p 2233 -v 0,1 --run_val --val_device 2
```

## Unrolls
The training regime from the Re3 paper is to start with 2 unrolls and a batch size of 64. When the loss plateaus, unroll 2x and decrease the batch size by 2x (or don't if you have enough memory). The loss will be written to the logs which can be read via running [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). I find this quite helpful. Every time you increase the number of unrolls, the loss will jump up. This is expected. I have also set up tensorboard to show images of the first 3 channels of each convolutional filter. This can be useful for debugging as well to make sure the initialization is correct, and that values are changing between iterations.

## Testing
To test the network, run the test.py script. This has many helpful flags which can tweak the testing, such as choosing to display or not display the images, to record a video, to skip some number of initial frames, to only test every n videos, and more. To view all the options, use the `-h` flag.


