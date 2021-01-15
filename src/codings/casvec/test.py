import math
import time
import numpy as np
import copy
import torch
from torch._C import device


# filepath = '/home/keke/Documents/Project/Sketch_Pytorch/resnet50109.npy'
#     # filepath = '/home/keke/Documents/Project/Sketch_Pytorch/attention56109.npy'

# gradients = ((np.load(filepath)))
# gradients = torch.from_numpy(gradients).to('cuda')
# for num in [4,8,16,32,64,128,256,512,1024]:
#     data = torch.zeros(gradients.size(), device='cuda')
#     start = time.time()
#     for i in range(num):
#         data+=gradients
#     end = time.time()
#     print(f"{num} time {end-start}")


bit_3_cplusd=np.array([0.0014595985412597656, 0.002256155014038086, 0.005036830902099609])

bit3_a=np.array([3.552436828613281e-05,6.437301635742188e-05,0.00012874603271484375])
bit3_find=0.0004

bit_2_cplusd=np.array([0.0009508132934570312, 0.0012049674987792969, 0.0023813247680664062] )
bit2_a=np.array([3.838539123535156e-05,6.723403930664062e-05,0.00013113021850585938])
bit2_find=0.0002

bit_1_cplusd=np.array([0.00043964385986328125, 0.0006885528564453125, 0.0013751983642578125]) 
bit1_a=np.array([3.910064697265625e-05,7.414817810058594e-05,0.00014781951904296875])
bit1_find=0.0001

data = []
for node in [4,8,16,32,64,128,256,512]:
    decode = node* ((bit_3_cplusd/np.array([4,8,16])).mean())
    merge = node* ((bit3_a/np.array([4,8,16])).mean())+bit3_find
    speedup = decode/merge
    data.append([merge, decode, speedup])
data = np.array(data)
print(data)
np.save('time_3bit.npy', data)

data = []
for node in [4,8,16,32,64,128,256,512]:
    decode = node* ((bit_2_cplusd/np.array([4,8,16])).mean())
    merge = node* ((bit2_a/np.array([4,8,16])).mean())+bit2_find
    speedup = decode/merge
    data.append([merge, decode, speedup])
data = np.array(data)
print(data)
np.save('time_2bit.npy', data)

data = []
for node in [4,8,16,32,64,128,256,512]:
    decode = node* ((bit_1_cplusd/np.array([4,8,16])).mean())
    merge = node* ((bit1_a/np.array([4,8,16])).mean())+bit1_find
    speedup = decode/merge
    data.append([merge, decode, speedup])
data = np.array(data)
print(data)
np.save('time_1bit.npy', data)