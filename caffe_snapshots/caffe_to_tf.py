import sys
import numpy as np
import tensorflow as tf
import glob
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    'tracker')))
import network

# Load tensorflow
tf.Graph().as_default()
batchSize = 1
delta = 1
imagePlaceholder = tf.placeholder(tf.float32, shape=(batchSize * delta * 2, 227, 227, 3))
labelsPlaceholder = tf.placeholder(tf.float32, shape=(batchSize * delta, 4))
learningRate = tf.placeholder(tf.float32)
tfOutputs = network.inference(imagePlaceholder, num_unrolls=delta, train=True)
tfLossFull, tfLoss = network.loss(tfOutputs, labelsPlaceholder)
train_op = network.training(tfLossFull, learningRate)
summary = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
summary_writer = tf.summary.FileWriter('logs/train/caffe_copy', sess.graph)
ops = []
with sess.as_default():
    sess.run(init)

    import caffe
    caffe.set_mode_cpu()
    # Load caffe net
    net_snapshot = sorted(glob.glob('*.caffemodel'), key=os.path.getmtime)[-1]
    reference_net = caffe.Net('RNNNet_deploy.prototxt', net_snapshot, 0)

    # Caffe, Tensorflow
    copy_params = [
            ('conv1', 'conv1'),
            ('conv1_reduce', 'conv1_skip'),
            ('relu1_reduce', 'conv1_skip/prelu'),
            ('conv2', 'conv2'),
            ('conv2_reduce', 'conv2_skip'),
            ('relu2_reduce', 'conv2_skip/prelu'),
            ('conv3', 'conv3'),
            ('conv4', 'conv4'),
            ('conv5', 'conv5'),
            ('conv5_reduce', 'conv5_skip'),
            ('relu5_reduce', 'conv5_skip/prelu'),
            ('fc6', 'fc6'),
            ('lstm1_block_00', 'lstm1/rnn/LSTM/block_input'),
            ('input1_gate_00', 'lstm1/rnn/LSTM/input_gate'),
            ('forget1_gate_00', 'lstm1/rnn/LSTM/forget_gate'),
            ('output1_gate_00', 'lstm1/rnn/LSTM/output_gate'),
            ('lstm2_block_00', 'lstm2/rnn/LSTM/block_input'),
            ('input2_gate_00', 'lstm2/rnn/LSTM/input_gate'),
            ('forget2_gate_00', 'lstm2/rnn/LSTM/forget_gate'),
            ('output2_gate_00', 'lstm2/rnn/LSTM/output_gate'),
            ('output_xyxy', 'fc_output'),
    ]

    tfVars = {var.name : var for var in tf.global_variables()}
    print('\n'.join(sorted([key for key in sorted(tfVars.keys()) if 'conv1' in key])) + '\n')
    for copy_param in copy_params:
        caffe_param = copy_param[0]
        # Weights
        if 'conv' in caffe_param:
            caffe_data = reference_net.params[caffe_param][0].data.transpose(2,3,1,0)
        else:
            caffe_data = reference_net.params[caffe_param][0].data
            if len(copy_param) > 2:
                # Have to deal with Caffe CxHxW vs Tensorflow HxWxC ordering
                tfShape = tf.get_default_graph().get_tensor_by_name(copy_param[2]).get_shape().as_list()
                caffe_data = caffe_data.reshape((caffe_data.shape[0], tfShape[3], tfShape[1], tfShape[2]))
                caffe_data = caffe_data.transpose(2,3,1,0)
                caffe_data = caffe_data.reshape(-1, caffe_data.shape[-1])
            else:
                caffe_data = caffe_data.T

        tf_param = 're3/' + copy_param[1]
        if 'conv' in copy_param[0]:
            tfVar = tfVars[tf_param + '/W_conv:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'weights to', tf_param)

            # Biases
            caffe_data = reference_net.params[caffe_param][1].data
            tfVar = tfVars[tf_param + '/b_conv:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'biases to', tf_param)
        elif 'fc' in copy_param[1]:
            tfVar = tfVars[tf_param + '/W_fc:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'weights to', tf_param)

            # Biases
            caffe_data = reference_net.params[caffe_param][1].data
            tfVar = tfVars[tf_param + '/b_fc:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'biases to', tf_param)
        elif 'lstm' in copy_param[1]:
            tfVar = tfVars[tf_param + '/weights:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'weights to', tf_param)

            # Biases
            caffe_data = reference_net.params[caffe_param][1].data
            tfVar = tfVars[tf_param + '/biases:0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'biases to', tf_param)
        else:
            tfVar = tfVars[tf_param + ':0']
            ops.append(tfVar.assign(caffe_data))
            print('copied ', caffe_param, 'to', tf_param)

    print('starting ops')
    sess.run(ops)
    print('done ops')
    print('starting summary')
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary_str, _ = sess.run([summary, train_op],
            feed_dict={
                imagePlaceholder : np.zeros((batchSize * delta * 2, 227, 227, 3)),
                labelsPlaceholder : np.zeros((batchSize, 4)),
                learningRate: 0,
                    },
                          options=run_options,
                          run_metadata=run_metadata)
    summary_writer.add_run_metadata(run_metadata, 'step_%07d' % 0)
    summary_writer.add_summary(summary_str, 0)
    summary_writer.flush()
    print('summary done')
    print('starting save')
    saveDir = 'logs/caffe_copy'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    checkpoint_file = os.path.join(saveDir, 'model.ckpt')
    saver.save(sess, checkpoint_file, global_step=0)
    print('saved successfully')

    IMAGENET_MEAN = np.array([123.151630838, 115.902882574, 103.062623801])

    random_input = np.random.random((2, 227, 227, 3))
    reference_net.blobs['image_data'].data[...] = (random_input).transpose((0,3,1,2))
    reference_net.blobs['image_data'].data[...] = (random_input - IMAGENET_MEAN).transpose((0,3,1,2))
    reference_net.forward()
    conv1Out = reference_net.blobs['conv1'].data

    conv1OutTF = sess.run(tf.get_default_graph().get_tensor_by_name('re3/conv1/Relu:0'), feed_dict={imagePlaceholder: random_input})
    conv1OutTF = conv1OutTF.transpose(0,3,1,2)

    print(conv1Out)
    print(conv1Out - conv1OutTF)
    print(np.max(np.abs(conv1Out - conv1OutTF)))
    print(np.mean(np.abs(conv1Out - conv1OutTF)))
    assert(np.mean(np.abs(conv1Out - conv1OutTF)) < 0.001)

    print('tested successfully')



