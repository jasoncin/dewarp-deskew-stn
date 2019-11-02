from transformer import spatial_transformer_network as stn
import numpy as np
import tensorflow as tf
from data_loader_queue import DataGenerator
import cv2

MODEL_DIR = 'checkpoints/'
path_list_train = "./train.lst"
path_list_val = "./val.lst"
c_dim = n_class = 1

print("Start loading data.")
data_provider = DataGenerator(path_list_train, path_list_val, n_class, 4)

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(MODEL_DIR + 'ResNet.model-6707.meta')
saver.restore(sess,tf.train.latest_checkpoint(MODEL_DIR))

graph = tf.get_default_graph()

print([n.name for n in tf.get_default_graph().as_graph_def().node])

theta = graph.get_tensor_by_name("network/stn_0/Tanh_1:0")
# print("Theta1 shape, ", theta.get_shape().as_list())
# theta = graph.get_tensor_by_name("network/stn_0/fully_connected_layer_2/weights:0")
input_tensor = graph.get_tensor_by_name("train_inputs:0")

idx = 0
for i in range(0, 2):
    batch_x, batch_y = data_provider.get_data('validation')
    train_feed_dict = {
        input_tensor: batch_x,
    }

    # theta = tf.eye(3, batch_shape=[4])
    # theta = tf.eye(num_rows=1, num_columns=9, batch_shape=[4])
    # theta = tf.reshape(theta, ([4, -1]))
    # print("Theta shape, ", theta.get_shape().as_list())

    logits = stn(input_tensor, theta)
    imgs = sess.run([logits], feed_dict=train_feed_dict)

    imgs = np.array(imgs)
    batch_y = np.array(batch_y)

    y_pred = imgs.flatten()
    y = batch_y.flatten()

    summation = 0
    n = len(y)
    for i in range(0, n):
        difference = y[i] - y_pred[i]
        squared_difference = difference ** 2
        summation = summation + squared_difference
    MSE = summation / n

    mse = np.sum(y - y_pred)
    print("MSE = {}".format(MSE))

    print(imgs.shape)
    for img in imgs[0]:
        print(img.shape)
        # img = 1.0 - img
        # img = img * 255
        cv2.imwrite("debug/tf_img_{}.png".format(idx), img)
        cv2.imshow("Output image", img)
        cv2.waitKey(0)
        print("Write image succesfully!")
        idx += 1
