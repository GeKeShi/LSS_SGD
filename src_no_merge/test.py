import torch

from mpi4py import MPI
import socket

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
torch.cuda.set_device(rank % 4)

print ('Hello world from process %d from %s.' % (rank, get_host_ip()), torch.Tensor(1,3).cuda())
# print(torch.__file__)
# print(torch.Tensor(2,4).cuda(rank % 4))
