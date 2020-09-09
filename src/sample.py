'''
======================================================
SMPL
======================================================
'''

from smplx.body_models import SMPL 
import torch
import torchvision.models as models 
import os 
import numpy as np

print(os.getcwd())

device = 'cuda:0' if torch.cuda.is_available else 'cpu'

smpl = SMPL('./SMPL_weight/basicModel_f_lbs_10_207_0_v1.1.0.pkl').to(device)

pose= np.array([
        1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
        -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
        -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
        1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
        2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
        7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
        -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
        -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
        -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
        9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
        -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
        -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
        -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
        -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
        -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
        3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
        -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
        6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
        -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
        4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
        2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
        -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
        -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
        3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)
    
beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
        0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])

vbeta = torch.tensor(np.array([beta])).float().to(device)
vpose = torch.tensor(np.array([pose])).float().to(device)

# print(f'vbeta device is {vbeta.device}')

output = smpl(vbeta, vpose)     # smpl input = (beta, theta)

resnet50 = models.resnet50(pretrained=True)

print(f'output vertices shape is {output.vertices.shape}')
print(f'output joints shape is {output.joints.shape}')
print(f'output betas shape is {output.betas.shape}')
print(f'output body_pose shape is {output.body_pose.shape}')        # 3 x 23(each part) + 3(root orientation)


'''
======================================================
Linear Model
======================================================
'''
import torch.nn as nn
import numpy as np
import sys
import torch

class LinearModel(nn.Module):
    '''
        input param:
            fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
            use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
            drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
            use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    '''
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers     = fc_layers
        self.use_dropout   = use_dropout
        self.drop_prob     = drop_prob
        self.use_ac_func   = use_ac_func
        
        if not self._check():
            msg = 'wrong LinearModel parameters!'
            print(msg)
            sys.exit(msg)

        self.create_layers()

    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                print('fc_layers require list, get {}'.format(type(self.fc_layers)))
                break
            
            if not isinstance(self.use_dropout, list):
                print('use_dropout require list, get {}'.format(type(self.use_dropout)))
                break

            if not isinstance(self.drop_prob, list):
                print('drop_prob require list, get {}'.format(type(self.drop_prob)))
                break

            if not isinstance(self.use_ac_func, list):
                print('use_ac_func require list, get {}'.format(type(self.use_ac_func)))
                break
            
            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_porb = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)

            return l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_porb < l_fc_layer and l_use_ac_func < l_fc_layer and l_drop_porb == l_use_drop

        return False

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()
        
        for i in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name = 'regressor_fc_{}'.format(i),
                module = nn.Linear(in_features = self.fc_layers[i], out_features = self.fc_layers[i + 1])
            )
            
            if i < l_use_ac_func and self.use_ac_func[i]:
                self.fc_blocks.add_module(
                    name = 'regressor_af_{}'.format(i),
                    module = nn.ReLU()
                )
            
            if i < l_use_drop and self.use_dropout[i]:
                self.fc_blocks.add_module(
                    name = 'regressor_fc_dropout_{}'.format(i),
                    module = nn.Dropout(p = self.drop_prob[i])
                )

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)

if __name__ == '__main__':
    fc_layers = [2133, 1024, 1024, 85]
    iterations = 3
    use_dropout = [True, True, False]
    drop_prob = [0.5, 0.5, 0]
    use_ac_func = [True, True, False]
    device = torch.device('cuda')
    net = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func).to(device)
    print(net)
    nx = np.zeros([2, 2048])
    vx = torch.from_numpy(nx).to(device)

