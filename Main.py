import os
import math
import random
import itertools
import torch 
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
from Database import Database
from CNP import CNP
from utility import get_context_target_tensor_3D, n_room_acc, room_confidence_acc

db = Database(n=1_000, fs=8_000, fv=1_000, n_mics=1000, name='db-fr-rs-rc', order=10)
train, validation = db()

cuda = torch.cuda.is_available()
path = "model.h5"

input_dim = 9
layers = (input_dim + 1, 128, 128, 128, 128)
cnp = CNP(encoder_layers=layers, decoder_layers=layers[1:], 
          input_dim=input_dim, dropout=0.2)
if cuda:
    cnp = cnp.cuda() 

cnp.init_weights()

if os.path.isfile(path):
    cnp.load_state_dict(torch.load(path), strict=False) # warm-starting a model (source: https://bit.ly/3IwhA9s)

optimizer = torch.optim.Adam(cnp.parameters(), lr=1E-3, weight_decay=5)

epochs = 10_000
batch_size = 32
sampling_rate = 0.5

(context_t, context_mask_t, target_x_t, target_y_t, target_mask_t) = get_context_target_tensor_3D(train, repeat=1, cuda=cuda, sampling_rate=sampling_rate)
(context_v, context_mask_v, target_x_v, target_y_v, target_mask_v) = get_context_target_tensor_3D(validation, repeat=1, cuda=cuda, sampling_rate=sampling_rate)

loss_list = []
val_loss_list = []

for i in tqdm(range(epochs)):
    for j in range(int(len(context_t)/batch_size)):
        s = random.randint(0, len(context_t)-batch_size)
        cnp(context_t[s:s+batch_size], context_mask_t[s:s+batch_size], target_x_t[s:s+batch_size])
        loss = cnp.loss(target_y_t[s:s+batch_size], target_mask_t[s:s+batch_size])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if i % 5 == 0:
        cnp(context_t, context_mask_t, target_x_t)
        loss = cnp.loss(target_y_t, target_mask_t)
        loss_list.append( (i, loss.item()) )
        cnp(context_v, context_mask_v, target_x_v)
        val_loss = cnp.loss(target_y_v, target_mask_v)
        val_loss_list.append( (i, val_loss.item()) ) 


epoch, train_acc = list(zip(*loss_list))
_, val_acc = list(zip(*val_loss_list))
plt.plot(epoch, train_acc, label="train")
plt.plot(epoch, val_acc, label="validation")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("loss.png")
plt.show()

torch.save(cnp.state_dict(), path)

n_room_acc(train[:5], cnp, n=5, plot=True)
n_room_acc(train[:30], cnp, n=30, plot=False)
n_room_acc(validation, cnp, plot=True)

room_confidence_acc(train[:30], cnp)
# room_confidence_acc(validation)