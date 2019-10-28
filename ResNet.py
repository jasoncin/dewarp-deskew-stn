import time
from ops import *
from utils import *
from data_loader_queue import DataGenerator

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset
        self.data_provider = None

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n
        self.epoch = args.epoch
        self.batch_size = args.batch_size

        if self.dataset_name == 'other':
            path_list_train = "./train.lst"
            path_list_val = "./val.lst"
            self.c_dim = n_class = 1

            print("Start loading data.")
            self.data_provider = DataGenerator(path_list_train, path_list_val, n_class, self.batch_size)
            self.img_h = self.data_provider.img_height
            self.img_w = self.data_provider.img_width

            with open(path_list_train, "r") as f:
                self.len_data = len(f.readlines())
                print(f.readlines())

        print("Length of data", self.len_data)
        print("End loading data.")

        self.iteration = int(self.len_data / self.batch_size)

        self.init_lr = args.lr


    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            B = x.get_shape().as_list()[0]
            H = x.get_shape().as_list()[1]
            W = x.get_shape().as_list()[2]
            C = x.get_shape().as_list()[3]

            x, self.stn_theta = spatial_transformer_layer("stn_0", input_tensor=x, img_size=[W, H], kernel_size=[3, 3, C, 100])

            #
            # # Add spatial transformer
            # with tf.variable_scope('spatial_transformer_0'):
            #     n_fc = 9
            #     W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
            #
            #     # %% Zoom into the image
            #     initial = np.array([[0.5, 0, 0], [0, 0.5, 0], [0.5, 0, 0]])
            #     initial = initial.astype('float32')
            #     initial = initial.flatten()
            #
            #     b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
            #     self.stn_theta = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1
            #
            #     x = stn(x, self.stn_theta)
            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_h, self.img_w, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.img_h, self.img_w, self.c_dim], name='train_labels')

        # self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_h, self.img_w, self.c_dim], name='test_inputs')
        # self.test_labels = tf.placeholder(tf.float32, [len(self.test_x), self.img_h, self.img_w, self.c_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        # self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        self.train_loss = classification_loss(labels=self.train_labels, theta=self.stn_theta, org=self.train_inptus)
        # self.test_loss = classification_loss(labels=self.test_labels, theta=self.stn_theta, org=self.test_inptus)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        # self.test_loss += reg_loss


        """ Training """
        # self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        # self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        # self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        # self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        # self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        # self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        could_load, checkpoint_counter = False, 0
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        min_loss = 10000000.0
        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                # batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x, batch_y = self.data_provider.get_data()
                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                # test_feed_dict = {
                #     self.test_inptus : self.test_x,
                #     self.test_labels : self.test_y
                # }


                # update network
                _, train_loss= self.sess.run(
                    [self.optim,  self.train_loss], feed_dict=train_feed_dict)
                # self.writer.add_summary(summary_str, counter)
                if min_loss >= train_loss:
                    min_loss = train_loss
                    self.save(self.checkpoint_dir, counter)
                # test
                # test_loss = self.sess.run(
                #     [ self.test_loss], feed_dict=test_feed_dict)
                # self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, learning_rate : %.4f, train_loss: %.4f, test_loss: %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, epoch_lr, train_loss, 0.01))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0



        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=2)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # test_feed_dict = {
        #     self.test_inptus: self.test_x,
        #     self.test_labels: self.test_y
        # }
        #
        #
        # test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        # print("test_accuracy: {}".format(test_accuracy))
