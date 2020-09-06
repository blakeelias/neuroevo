import mkl
mkl.set_num_threads(1)
print('Successfully imported mkl...')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

def step(env, *args):
    state, a, b, c = env.step(*args)
    state = convert_state(state)
    return state, a, b, c

def reset(env):
    return convert_state(env.reset())

def convert_state(state):
    import cv2
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TODO: padding?
        self.conv1 = nn.Conv2d(4, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.dense = nn.Linear(4*4*64, 512)
        self.out = nn.Linear(512, 18)
                
        torch.manual_seed(random_state())
                
        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal(tensor)
            else:
                tensor.data.zero_()
                        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        return self.out(x)
    
    def evolve(self, sigma):
        torch.manual_seed(random_state())
                    
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)
            
def random_state():
    return random.randint(0, 2**31-1)
    
def evaluate_model(env, model, max_eval=20000, max_noop=30):
    import gym
    env = gym.make(env)
    noops = random.randint(0, max_noop)
    cur_states = [reset(env)] * 4
    total_reward = 0
    for _ in range(noops):
        cur_states.pop(0)
        new_state, reward, is_done, _ = step(env, 0)
        total_reward += reward
        if is_done:
            return total_reward
        cur_states.append(new_state)

    total_frames = 0
    model.eval()
    for _ in range(max_eval):
        total_frames += 4
        values = model(Variable(torch.Tensor([cur_states])))[0]
        action = np.argmax(values.data.numpy()[:env.action_space.n])
        # print(action)
        new_state, reward, is_done, _ = step(env, action)
        total_reward += reward
        if is_done:
            break
        cur_states.pop(0)
        cur_states.append(new_state)

    print('\t', total_reward)
    return total_reward, total_frames
