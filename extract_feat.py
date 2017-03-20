# TensorFlow version of NIPS2016 soundnet
# Required package: librosa: A python package for music and audio analysis.
# $ pip install librosa

import tensorflow as tf
import numpy as np
import librosa
import pdb
import sys

config = {  'batch_size': 1, 
            'eps': 1e-5,
            'sample_rate': 22050,
            }

class Model():
    def __init__(self, config, param_G=None):
        self.config = config
        self.param_G = param_G
        self.add_placeholders()
        self.add_generator()


    def add_placeholders(self):
        self.sound_input_placeholder = tf.placeholder(tf.float32,
                shape=[self.config['batch_size'], None, 1, 1]) # batch x h x w x channel


    def add_generator(self):
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
        self.layers[25] = self.conv2d(self.layers[24], 1024, 1000, k_h=8, d_h=2, name_scope='conv8')
        self.layers[26] = self.conv2d(self.layers[24], 1024, 401, k_h=8, d_h=2, name_scope='conv8_2')


    def conv2d(self, prev_layer, in_ch, out_ch, k_h=1, k_w=1, d_h=1, d_w=1, p_h=0, p_w=0, name_scope='conv'):
        with tf.variable_scope(name_scope) as scope:
            # h x w x input_channel x output_channel
            w_conv = tf.get_variable('weights', [k_h, k_w, in_ch, out_ch], 
                    initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
            b_conv = tf.get_variable('biases', [out_ch], 
                    initializer=tf.constant_initializer(0.0))
            
            padded_input = tf.pad(prev_layer, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], "CONSTANT")

            output = tf.nn.conv2d(padded_input, w_conv, 
                    [1, d_h, d_w, 1], padding='VALID', name='z') + b_conv
        
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


    def load(self, sess):
        if self.param_G is not None:
            data_dict = self.param_G
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            sess.run(var.assign(data_dict[key][subkey]))
                            print 'Assign pretrain model ' + subkey + ' to ' + key
                        except:
                            print 'Ignore ' + key


def preprocess(raw_audio, batch_size):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[:, 0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we cannnot pick the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to BatchSize x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [batch_size, -1, 1, 1])

    return raw_audio


def extract_feat(sound_input, layer_min, layer_max=None):
    # Load pre-trained model
    G_name = './models/sound8.npy'
    param_G = np.load(G_name).item()
    dump_path = './output/'

    # Build model
    model = Model(config, param_G)
    init = tf.global_variables_initializer()

    feed_dict = {model.sound_input_placeholder: sound_input}

    # Extract feature
    if layer_max is None:
        layer_max = layer_min + 1    
   
    with tf.Session() as session:
        session.run(init)
        model.load(session)
        
        for idx in xrange(layer_min, layer_max):
            feature = session.run(model.layers[idx], feed_dict=feed_dict)
            if (idx - layer_min) == 0: pdb.set_trace()
            np.save(dump_path + 'tf_fea{}.npy'.format(idx), np.squeeze(feature))
            print "Save layer {} with shape {} as {}tf_fea{}.npy".format(idx, np.squeeze(feature).shape, dump_path, idx)
    
    return feature


if __name__ == '__main__':

    layer_min = int(sys.argv[1])
    layer_max = int(sys.argv[2]) if len(sys.argv) > 2 else layer_min + 1

    # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
    audio_path = './data/demo.mp3'
    sound_sample, _ = librosa.load(audio_path, sr=None, mono=True)
    sound_sample = preprocess(sound_sample, config['batch_size'])
    
    # Demo
    sound_sample = np.reshape(np.load('data/demo.npy'), [config['batch_size'], -1, 1, 1])
    
    output = extract_feat(sound_sample, layer_min, layer_max=layer_max)


