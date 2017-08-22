# Load t7 files
# Required package: torchfile. 
# $ pip install torchfile

import torchfile
import numpy as np
import pdb

# Make xrange compatible in both Python 2, 3
try:
    xrange
except NameError:
    xrange = range

keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6',
        'conv7', 'conv8', 'conv8_2']

def load(o, param_list):
    """ Get torch7 weights into numpy array """
    try:
        num = len(o['modules'])
    except:
        num = 0
    
    for i in xrange(num):
        # 2D conv
        if o['modules'][i]._typename == 'nn.SpatialConvolution' or \
            o['modules'][i]._typename == 'cudnn.SpatialConvolution':
            temp = {'weights': o['modules'][i]['weight'].transpose((2,3,1,0)),
                    'biases': o['modules'][i]['bias']}
            param_list.append(temp)
        # 2D deconv
        elif o['modules'][i]._typename == 'nn.SpatialFullConvolution':
            temp = {'weights': o['modules'][i]['weight'].transpose((2,3,1,0)),
                    'biases': o['modules'][i]['bias']}
            param_list.append(temp)
        # 3D conv
        elif o['modules'][i]._typename == 'nn.VolumetricFullConvolution':
            temp = {'weights': o['modules'][i]['weight'].transpose((2,3,4,1,0)),
                    'biases': o['modules'][i]['bias']}
            param_list.append(temp)
        # batch norm
        elif o['modules'][i]._typename == 'nn.SpatialBatchNormalization' or \
            o['modules'][i]._typename == 'nn.VolumetricBatchNormalization':
            param_list[-1]['gamma'] = o['modules'][i]['weight']
            param_list[-1]['beta'] = o['modules'][i]['bias']
            param_list[-1]['mean'] = o['modules'][i]['running_mean']
            param_list[-1]['var'] = o['modules'][i]['running_var']

        load(o['modules'][i], param_list)


def show(o):
    """ Show nn information """
    nn = {}
    nn_keys = {}
    nn_info = {}
    num = len(o['modules']) if o['modules'] else 0
    mylist = get_mylist()

    for i in xrange(num):
        # Get _obj and keys from torchfile
        nn[i] = o['modules'][i]._obj
        nn_keys[i] = o['modules'][i]._obj.keys()
        
        # Get information from _obj
        # {layer i: {mylist keys: value}}
        nn_info[i] = {key: nn[i][key] for key in sorted(nn_keys[i]) if key in mylist}
        nn_info[i]['name'] = o['modules'][i]._typename
        print(i, nn_info[i]['name'])
        for item in sorted(nn_info[i].keys()): 
            print("  {}:{}".format(item, nn_info[i][item] if 'running' not in item \
                                                        else nn_info[i][item].shape))


def get_mylist():
    """ Return manually selected information lists """
    return ['_type', 'nInputPlane', 'nOutputPlane', \
            'input_offset', 'groups', 'dH', 'dW', \
            'padH', 'padW', 'kH', 'kW', 'iSize', \
            'running_mean', 'running_var']


if __name__ == '__main__':
    # File loader
    t7_file = './models/soundnet8_final.t7'
    o = torchfile.load(t7_file)
    
    # To show nn parameter
    show(o)
    
    # To store as npy file
    param_list = []
    load(o, param_list)
    save_list = {}
    for i, k in enumerate(keys):
        save_list[k] = param_list[i]
    np.save('sound8', save_list)

