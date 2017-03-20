# TensorFlow version of NIPS2016 soundnet
# Required package: librosa: A python package for music and audio analysis.
# $ pip install librosa

import tensorflow as tf
from glob import glob
import numpy as np
import librosa
import time
import pdb
import sys
import os

config = {  'batch_size': 1, 
            'train_size': np.inf,
            'epoch': 200,
            'eps': 1e-5,
            'learning_rate': 1e-3,
            'beta1': 0.9,
            'load_size': 22050*4,
            'sample_rate': 22050,
            'name_scope': 'SoundNet',
            'dataset_name': 'ESC50',
            'subname': 'mp3',
            'checkpoint_dir': 'checkpoint',
            'dump_dir': 'output',
            'model_dir': None,
            'param_g_dir': None,
            'phase': 'train', # 'train', 'extract'
            }


class Model():
    def __init__(self, session, config, param_G=None):
        self.sess           = session
        self.config         = config
        self.param_G        = param_G
        self.g_step         = tf.Variable(0, trainable=False)
        self.counter        = 0
        self.model()
 

    def model(self):
        # Placeholder
        self.sound_input_placeholder = tf.placeholder(tf.float32,
                shape=[self.config['batch_size'], None, 1, 1]) # batch x h x w x channel
        self.object_dist = tf.placeholder(tf.float32,
                shape=[self.config['batch_size'], None, 1000]) # batch x h x w x channel
        self.scene_dist = tf.placeholder(tf.float32,
                shape=[self.config['batch_size'], None, 401]) # batch x h x w x channel
        
        # Generator
        self.add_generator(name_scope=self.config['name_scope'])
 
        # KL Divergence
        self.object_loss = self.KL_divergence(self.layers[25], self.object_dist, name_scope='KL_Div_object')
        self.scene_loss = self.KL_divergence(self.layers[26], self.scene_dist, name_scope='KL_Div_scene')
        self.loss = self.object_loss + self.scene_loss

        # Summary
        self.loss_sum = tf.summary.scalar("g_loss", self.loss)
        self.g_sum = tf.summary.merge([self.loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        
        # variable collection
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                    scope=self.config['name_scope'])

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=12, 
                                    max_to_keep=5, 
                                    restore_sequentially=True)
        

        # Optimizer and summary
        self.g_optim = tf.train.AdamOptimizer(self.config['learning_rate'], beta1=self.config['beta1']) \
                          .minimize(self.loss, var_list=(self.g_vars), global_step=self.g_step)
        
        # Initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        # Load checkpoint
        if self.load(self.config['checkpoint_dir']):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


    def add_generator(self, name_scope='SoundNet'):
        with tf.variable_scope(name_scope) as scope:
            self.layers = {}
            
            # Stream one: conv1 ~ conv7
            self.layers[1] = self.conv2d(self.sound_input_placeholder, 1, 16, k_h=64, d_h=2, p_h=32, name_scope='conv1')
            self.layers[2] = self.batch_norm(self.layers[1], 16, self.config['eps'], name_scope='conv1')
            self.layers[3] = self.relu(self.layers[2], name_scope='conv1')
            self.layers[4] = self.maxpool(self.layers[3], k_h=8, d_h=8, name_scope='conv1')

            self.layers[5] = self.conv2d(self.layers[4], 16, 32, k_h=32, d_h=2, p_h=16, name_scope='conv2')
            self.layers[6] = self.batch_norm(self.layers[5], 32, self.config['eps'], name_scope='conv2')
            self.layers[7] = self.relu(self.layers[6], name_scope='conv2')
            self.layers[8] = self.maxpool(self.layers[7], k_h=8, d_h=8, name_scope='conv2')

            self.layers[9] = self.conv2d(self.layers[8], 32, 64, k_h=16, d_h=2, p_h=8, name_scope='conv3')
            self.layers[10] = self.batch_norm(self.layers[9], 64, self.config['eps'], name_scope='conv3')
            self.layers[11] = self.relu(self.layers[10], name_scope='conv3')

            self.layers[12] = self.conv2d(self.layers[11], 64, 128, k_h=8, d_h=2, p_h=4, name_scope='conv4')
            self.layers[13] = self.batch_norm(self.layers[12], 128, self.config['eps'], name_scope='conv4')
            self.layers[14] = self.relu(self.layers[13], name_scope='conv4')

            self.layers[15] = self.conv2d(self.layers[14], 128, 256, k_h=4, d_h=2, p_h=2, name_scope='conv5')
            self.layers[16] = self.batch_norm(self.layers[15], 256, self.config['eps'], name_scope='conv5')
            self.layers[17] = self.relu(self.layers[16], name_scope='conv5')
            self.layers[18] = self.maxpool(self.layers[17], k_h=4, d_h=4, name_scope='conv5')

            self.layers[19] = self.conv2d(self.layers[18], 256, 512, k_h=4, d_h=2, p_h=2, name_scope='conv6')
            self.layers[20] = self.batch_norm(self.layers[19], 512, self.config['eps'], name_scope='conv6')
            self.layers[21] = self.relu(self.layers[20], name_scope='conv6')

            self.layers[22] = self.conv2d(self.layers[21], 512, 1024, k_h=4, d_h=2, p_h=2, name_scope='conv7')
            self.layers[23] = self.batch_norm(self.layers[22], 1024, self.config['eps'], name_scope='conv7')
            self.layers[24] = self.relu(self.layers[23], name_scope='conv7')

            # Split one: conv8, conv8_2
            # NOTE: here we use a padding of 2 to skip an unknown error
            # https://github.com/dennybritz/cnn-text-classification-tf/issues/8#issuecomment-209528768
            self.layers[25] = self.conv2d(self.layers[24], 1024, 1000, k_h=8, d_h=2, p_h=2, name_scope='conv8')
            self.layers[26] = self.conv2d(self.layers[24], 1024, 401, k_h=8, d_h=2, p_h=2, name_scope='conv8_2')


    def train(self):
        """Train SoundNet"""

        start_time = time.time()

        # Data info
        data = glob('./data/*.{}'.format(self.config['subname']))
        batch_idxs = min(len(data), self.config['train_size']) // self.config['batch_size']
        for epoch in xrange(self.counter//batch_idxs, self.config['epoch']):

            for idx in xrange(self.counter%batch_idxs, batch_idxs):
        
                # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
                sound_sample, _ = librosa.load(data[idx], sr=22050, mono=True)
                
                # Update G network
                # NOTE: Here we still use dummy distribution for scene and objects
                _, summary_str, l_scn, l_obj = self.sess.run([self.g_optim, self.g_sum, self.scene_loss, self.object_loss],
                    feed_dict={self.sound_input_placeholder: model.preprocess(sound_sample, length=self.config['load_size']), \
                            self.scene_dist: np.random.randint(2, size=(1, 1, 401)), \
                            self.object_dist: np.random.randint(2, size=(1, 1, 1000))})
                self.writer.add_summary(summary_str, self.counter)

                print "[Epoch {}] {}/{} : scene_loss: {} obj_loss: {}".format(epoch, idx, batch_idxs, l_scn, l_obj)

                if np.mod(self.counter, 1000) == 1000 - 1:
                    self.save(self.config['checkpoint_dir'], self.counter)

                self.counter += 1


    #########################
    #           OPs         #
    #########################
    def conv2d(self, prev_layer, in_ch, out_ch, k_h=1, k_w=1, d_h=1, d_w=1, p_h=0, p_w=0, pad='VALID', name_scope='conv'):
        with tf.variable_scope(name_scope) as scope:
            # h x w x input_channel x output_channel
            w_conv = tf.get_variable('weights', [k_h, k_w, in_ch, out_ch], 
                    initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv = tf.get_variable('biases', [out_ch], 
                    initializer=tf.constant_initializer(0.0))
            
            padded_input = tf.pad(prev_layer, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], "CONSTANT") if pad == 'VALID' \
                    else prev_layer

            output = tf.nn.conv2d(padded_input, w_conv, 
                    [1, d_h, d_w, 1], padding=pad, name='z') + b_conv
        
            return output


    def batch_norm(self, prev_layer, out_ch, eps, name_scope='conv'):
        with tf.variable_scope(name_scope) as scope:
            #mu_conv, var_conv = tf.nn.moments(prev_layer, [0, 1, 2], keep_dims=False)
            mu_conv = tf.get_variable('mean', [out_ch], 
            	initializer=tf.constant_initializer(0))
            var_conv = tf.get_variable('var', [out_ch], 
             	initializer=tf.constant_initializer(1))
            gamma_conv = tf.get_variable('gamma', [out_ch], 
            	initializer=tf.constant_initializer(1))
            beta_conv = tf.get_variable('beta', [out_ch], 
             	initializer=tf.constant_initializer(0))
            output = tf.nn.batch_normalization(prev_layer, mu_conv, 
             	var_conv, beta_conv, gamma_conv, eps, name='batch_norm')
            
            return output


    def relu(self, prev_layer, name_scope='conv'):
        with tf.variable_scope(name_scope) as scope:
            return tf.nn.relu(prev_layer, name='a')


    def maxpool(self, prev_layer, k_h=1, k_w=1, d_h=1, d_w=1, name_scope='conv'):
        with tf.variable_scope(name_scope) as scope:
            return tf.nn.max_pool(prev_layer, 
                    [1, k_h, k_w, 1], [1, d_h, d_w, 1], padding='VALID', name='maxpool')

    
    #########################
    #          Loss         #
    #########################
    # Use the answer here: http://stackoverflow.com/questions/41863814/kl-divergence-in-tensorflow
    def KL_divergence(self, dist_a, dist_b, name_scope='KL_Div'):
        return tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(dist_a, dist_b))


    #########################
    #       Save/Load       #
    #########################
    @property
    def get_model_dir(self):
        if self.config['model_dir'] is None:
            return "{}_{}".format(
                self.config['dataset_name'], self.config['batch_size'])
        else:
            return self.config['model_dir']
    

    def load(self, ckpt_dir='checkpoint'):
        return self.load_from_ckpt(ckpt_dir) if self.param_G is None \
        else self.load_from_npy()


    def save(self, checkpoint_dir, step):
        """ Checkpoint saver """
        model_name = "SoundNet.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load_from_ckpt(self, checkpoint_dir='checkpoint'):
        """ Checkpoint loader """
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            self.counter = int(ckpt_name.rsplit('-', 1)[-1])
            print(" [*] Start counter from {}".format(self.counter))
            return True
        else:
            print(" [*] Failed to find a checkpoint under {}".format(checkpoint_dir))
            return False


    def load_from_npy(self):
        if self.param_G is None: return False
        data_dict = self.param_G
        for key in data_dict:
            with tf.variable_scope(self.config['name_scope'] + '/'+ key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print 'Assign pretrain model ' + subkey + ' to ' + key
                    except:
                        print 'Ignore ' + key
        
        self.param_G.clear()
        return True


    #########################
    #      Preprocess       #
    #########################
    def preprocess(self, raw_audio, length=None):
        # Select first channel (mono)
        if len(raw_audio.shape) > 1:
            raw_audio = raw_audio[:, 0]

        # Make range [-256, 256]
        raw_audio *= 256.0

        # Use length or Not
        if length is not None:
            raw_audio = raw_audio[:length]

        # Check conditions
        assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we cannnot pick the first channel"
        assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
        assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

        # Shape to BatchSize x DIM x 1 x 1
        raw_audio = np.reshape(raw_audio, [self.config['batch_size'], -1, 1, 1])

        return raw_audio


if __name__ == '__main__':

    # Print config
    for key in config: print key, ":", config[key]

    # Load pre-trained model
    # TODO: make it switchable from ckpt to npy
    if config['param_g_dir'] is None:
        param_G = np.load('./models/sound8.npy').item()
        #param_G = None
    else:
        param_G = np.load(config['param_g_dir']).item()

    # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
    audio_path = './data/demo.mp3'
    sound_sample, _ = librosa.load(audio_path, sr=None, mono=True)

    with tf.Session() as session:
        # Build model
        model = Model(session, config, param_G)
 
        if config['phase'] == 'train':
            # Training phase
            model.train()
        elif config['phase'] == 'extract':
            # Feature extractor
            idx = int(sys.argv[1])
            demo = np.reshape(np.load('demo.npy'), [config['batch_size'], -1, 1, 1])
            feed_dict = {model.sound_input_placeholder: demo}
            feature = session.run(model.layers[idx], feed_dict=feed_dict)
            pdb.set_trace()
            np.save(os.path.join(config['dump_dir'], 'tf_fea{}.npy'.format(idx)), np.squeeze(feature))
            print "Save layer {} with shape {} as {}/tf_fea{}.npy".format(idx, np.squeeze(feature).shape, config['dump_dir'], idx)

