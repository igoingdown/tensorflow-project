#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   在分布式环境下实现tensorflow的一个小demo（手写数字识别，MNIST）。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# ps_hosts和worker_hosts的ip地址并不是固定的，
# 我在参数设置写死是因为我生成的docker container的ip地址是这样分布的。
# 端口号可以任意指定，只要是空闲端口号就行。
tf.app.flags.DEFINE_string("ps_hosts", 
                           "172.17.0.5:2222,172.17.0.4:2222", 
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", 
                           "172.17.0.3:2222,172.17.0.2:2222", 
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, 
                                                           FLAGS.task_index, 
                                                           server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model ...
            mnist = input_data.read_data_sets("data", one_hot=True)
            
            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(x, W) + b

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(y, y_), 
                    name="loss_op")
            # loss_summary = tf.summary.scalar("loss", cross_entropy)

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                cross_entropy, global_step=global_step)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), 
                                          tf.argmax(y_, 1), 
                                          name="correct_judge_op")
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
                                              tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy_summary", 
                                                 accuracy)
  
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        if not os.path.exists("mnist_log"):
            os.mkdir("mnist_log")
        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="mnist_log",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver = saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization 
        # and restoring from a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)
        writer = tf.summary.FileWriter("mnist_log", sess.graph)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while not sv.should_stop() and step < 10000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, step = sess.run([train_op, global_step], 
                               feed_dict={x: batch_xs, y_: batch_ys})
            if step % 100 == 0 and FLAGS.task_index != 0:
                res = sess.run(summary_op, 
                               feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels})
                writer.add_summary(res, step)
                print("Step {0} in task {1}".format(step, FLAGS.task_index))

        print("done.")
        if not os.path.exists("variables_saved"):
            os.mkdir("variables_saved")
        saver.save(sess, "variables_saved/variables")
        if FLAGS.task_index != 0:
            print("accuracy: %f" % sess.run(accuracy, 
                                            feed_dict={x: mnist.test.images,
                                                       y_: mnist.test.labels}))

if __name__ == "__main__":
    tf.app.run()



















