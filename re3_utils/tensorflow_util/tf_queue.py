import tensorflow as tf
import threading
import numpy as np
import time

DEBUG = True

class TFQueue(object):

    def __init__(self, sess, placeholders, max_queue_size, max_queue_uses, use_random_order, batch_size):
        self.sess = sess
        self.placeholders = placeholders
        self.max_queue_size = max_queue_size
        self.max_queue_uses = max_queue_uses
        self.data_buffer = []
        self.data_counts = np.zeros(max_queue_size)
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.enqueue_batch_size = self.placeholders[0].get_shape().as_list()[0]
        self.use_random_order = use_random_order
        self.num_samples = 0

        # Set up queue and operations
        with tf.device('/cpu:0'):
            self.queue = tf.FIFOQueue(self.max_queue_size,
                    [placeholder.dtype for placeholder in self.placeholders],
                    shapes=[placeholder.get_shape().as_list()[1:] for placeholder in self.placeholders])
            self.enqueue_op = self.queue.enqueue_many(self.placeholders)
            self.placeholder_outs = {self.placeholders[ii] : val for ii,val in enumerate(self.queue.dequeue_many(self.batch_size))}
            self.size = self.queue.size()

        # Start thread
        self.thread = threading.Thread(target=self.tf_enqueue_data)
        self.thread.daemon = True
        self.thread.start()

    def enqueue(self, data, lock=True):
        if lock:
            self.lock.acquire()
        if len(self.data_buffer) < self.max_queue_size:
            self.data_buffer.append(data)
        else:
            while np.max(self.data_counts) == 0:
                time.sleep(1)
            max_count_ind = np.argmax(self.data_counts)
            self.data_buffer[max_count_ind] = data
            self.data_counts[max_count_ind] = 0
        if lock:
            self.lock.release()

    def enqueue_many(self, feed_dict):
        self.lock.acquire()
        items = list(feed_dict.items())
        num_items = len(items[0][1])
        for ii in range(num_items):
            local_feed_dict = {key : val[ii] for key,val in items}
            self.enqueue(local_feed_dict, lock=False)
        self.lock.release()

    def get_feed_dict(self):
        self.lock.acquire()
        if self.use_random_order:
            if self.max_queue_uses > 0:
                usable_inds = np.where(self.data_counts[:len(self.data_buffer)] < self.max_queue_uses)[0]
            else:
                usable_inds = np.arange(len(self.data_buffer))
            chosen_inds = np.random.choice(usable_inds, self.enqueue_batch_size, replace=False)
        else:
            chosen_inds = np.lexsort((np.random.random(len(self.data_buffer)), self.data_counts[:len(self.data_buffer)]))[:self.enqueue_batch_size]

        self.data_counts[chosen_inds] += 1
        feed_dict = {placeholder : [] for placeholder in self.placeholders}
        for ind in chosen_inds:
            data = self.data_buffer[ind]
            for placeholder in self.placeholders:
                feed_dict[placeholder].append(data[placeholder])

        feed_dict = {key : np.ascontiguousarray(val) for (key, val) in feed_dict.items()}
        self.num_samples += 1
        if DEBUG and self.num_samples % 10 == 0:
            if len(self.data_buffer) < self.max_queue_size:
                print('Buffer size: %d  Num unused: %d  Max times used: %d Median times used: %d\n' % (
                        len(self.data_buffer),
                        (len(self.data_buffer) - len(self.data_counts[self.data_counts > 0])),
                        np.max(self.data_counts),
                        np.median(self.data_counts[:len(self.data_buffer)])))
            else:
                print('Buffer Full. Num unused: %d  Max times used: %d  Median times used: %d\n' % (
                        (len(self.data_buffer) - len(self.data_counts[self.data_counts > 0])),
                        np.max(self.data_counts),
                        np.median(self.data_counts)))


        self.lock.release()
        return feed_dict

    def tf_enqueue_data(self):
        while True:
            if self.max_queue_uses > 0:
                data_counts_curr_length = self.data_counts[:len(self.data_buffer)]
                while len(data_counts_curr_length[data_counts_curr_length < self.max_queue_uses]) < self.enqueue_batch_size:
                    time.sleep(1)
                    data_counts_curr_length = self.data_counts[:len(self.data_buffer)]
            else:
                while len(self.data_buffer) < min(10 * self.enqueue_batch_size, self.max_queue_size):
                    time.sleep(1)

            feed_dict = self.get_feed_dict()
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)

