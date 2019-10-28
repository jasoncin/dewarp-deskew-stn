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
data_provider = DataGenerator(path_list_train, path_list_val, n_class, 2)

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(MODEL_DIR + 'ResNet.model-2836.meta')
saver.restore(sess,tf.train.latest_checkpoint(MODEL_DIR))

graph = tf.get_default_graph()

print([n.name for n in tf.get_default_graph().as_graph_def().node])

theta = graph.get_tensor_by_name("network/stn_0/Tanh_1:0")
input_tensor = graph.get_tensor_by_name("train_inputs:0")

idx = 0
for i in range(0, 1):
    batch_x, batch_y = data_provider.get_data()
    train_feed_dict = {
        input_tensor: batch_x,
    }

    logits = stn(input_tensor, theta)
    imgs = sess.run(
        [logits], feed_dict=train_feed_dict)

    imgs = np.array(imgs)
    print(imgs.shape)
    for img in imgs[0]:
        print(img.shape)
        img = img * 255
        cv2.imwrite("tf_img_{}.png".format(idx), img)
        print("Write image succesfully!")
        idx += 1
