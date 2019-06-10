# Atomo: Communication-efficient Learning via Atomic Sparsification
This repository contains source code for Atomo, a general framework for atomic sparsification of stochastic gradients. Please check [the full paper](http://papers.nips.cc/paper/8191-atomo-communication-efficient-learning-via-atomic-sparsification) for detailed information about this project.

## Overview:
ATOMO is a general framework for atomic sparsification of stochastic gradients. Given a gradient, an atomic decomposition,
and a sparsity budget, ATOMO gives a random unbiased sparsification of the atoms minimizing variance. ATOMO sets up and optimally solves a meta-optimization that minimizes the variance of the sparsified gradient, subject to the constraints
that it is sparse on the atomic basis, and also is an unbiased estimator of the input.

<div align="center"><img src="https://github.com/hwang595/ATOMO/blob/master/images/SVdecay.jpg" height="350" width="450" ></div>

## Depdendencies:
Tested stable depdencises:
* python 2.7 (Anaconda)
* PyTorch 0.3.0 (*please note that, we're moving to PyTorch 0.4.0, and 1.0.x*)
* torchvision 0.1.18
* MPI4Py 0.3.0
* python-blosc 1.5.0

We highly recommend installing an [Anaconda](https://www.continuum.io/downloads) environment.
You will get a high-quality BLAS library (MKL) and you get a controlled compiler version regardless of your Linux distro.

We provide [this script](https://github.com/hwang595/ATOMO/blob/master/tools/pre_run.sh) to help you with building all dependencies. To do that you can run:
```
bash ./tools/pre_run.sh
```

## Cluster Setup:
For running on distributed cluster, the first thing you need do is to launch AWS EC2 instances.
### Launching Instances:
[This script](https://github.com/hwang595/ps_pytorch/blob/master/tools/pytorch_ec2.py) helps you to launch EC2 instances automatically, but before running this script, you should follow [the instruction](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to setup AWS CLI on your local machine.
After that, please edit this part in `./tools/pytorch_ec2.py`
``` python
cfg = Cfg({
    "name" : "PS_PYTORCH",      # Unique name for this specific configuration
    "key_name": "NameOfKeyFile",          # Necessary to ssh into created instances
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 8,
    "num_replicas_to_aggregate" : "8", # deprecated, not necessary
    "method" : "spot",
    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",
    # Machine type - instance type configuration.
    "master_type" : "m4.2xlarge",
    "worker_type" : "m4.2xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-xxxxxxxx",            # id of AMI
    # Launch specifications
    "spot_price" : "0.15",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/dir/to/NameOfKeyFile.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.14.225",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
```
For setting everything up on EC2 cluster, the easiest way is to setup one machine and create an AMI. Then use the AMI id for `image_id` in `pytorch_ec2.py`. Then, launch EC2 instances by running
```
python ./tools/pytorch_ec2.py launch
```
After all launched instances are ready (this may take a while), getting private ips of instances by
```
python ./tools/pytorch_ec2.py get_hosts
```
this will write ips into a file named `hosts_address`, which looks like
```
172.31.16.226 (${PS_IP})
172.31.27.245
172.31.29.131
172.31.18.108
172.31.18.174
172.31.17.228
172.31.16.25
172.31.30.61
172.31.29.30
```
After generating the `hosts_address` of all EC2 instances, running the following command will copy your keyfile to the parameter server (PS) instance whose address is always the first one in `hosts_address`. `local_script.sh` will also do some basic configurations e.g. clone this git repo
```
bash ./tool/local_script.sh ${PS_IP}
```
### SSH related:
At this stage, you should ssh to the PS instance and all operation should happen on PS. In PS setting, PS should be able to ssh to any compute node, [this part](https://github.com/hwang595/ATOMO/blob/master/tools/remote_script.sh#L8-L16) dose the job for you by running (after ssh to the PS)
```
bash ./tools/remote_script.sh
```

## Prepare Datasets
We currently support [MNIST](http://yann.lecun.com/exdb/mnist/) and [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Download, split, and transform datasets by (and `./tools/remote_script.sh` dose this for you)
```
bash ./src/data_prepare.sh
```

## Job Launching
Since this project is built on MPI, tasks are required to be launched by PS (or master) instance. `run_pytorch.sh` wraps job-launching process up. Commonly used options (arguments) are listed as following:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `n`                     | Number of processes (size of cluster) e.g. if we have P compute node and 1 PS, n=P+1. |
| `hostfile`      | A directory to the file that contains Private IPs of every node in the cluster, we use `hosts_address` here as [mentioned before](#launching-instances). |
| `lr` | Inital learning rate that will be use. |
| `momentum` | Value of momentum that will be use. |
| `network` | Types of deep neural nets, currently `LeNet`, `ResNet-18/32/50/110/152`, and `VGGs` are supported. |
| `dataset` | Datasets use for training. |
| `batch-size` | Batch size for optimization algorithms. |
| `test-batch-size` | Batch size used during model evaluation. |
| `comm-type` | A fake parameter, please always set it to be `Bcast`. |
| `num-aggregate` | Number of gradients required for the PS to aggregate. |
| `max-steps` | The maximum number of iterations to train. |
| `svd-rank`  | The expected rank of gradients ATOMO used (which is the same as the sparsity budget `s` in our paper). |
| `epochs`    | The maximal number of epochs to train (somehow redundant).   |
| `eval-freq` | Frequency of iterations to evaluation the model. |
| `enable-gpu`| Training on CPU/GPU, if CPU please leave this argument empty. |
|`train-dir`  | Directory to save model checkpoints for evaluation. |

## Model Evaluation
[Distributed evaluator](https://github.com/hwang595/ATOMO/blob/master/src/distributed_evaluator.py) will fetch model checkpoints from the shared directory and evaluate model on validation set.
To evaluate model, you can run
```
bash ./src/evaluate_pytorch.sh
```
with specified arguments.

Evaluation arguments are listed as following:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `eval-batch-size`             | Batch size (on validation set) used during model evaluation. |
| `eval-freq`      | Frequency of iterations to evaluation the model, should be set to the same value as [run_pytorch.sh](https://github.com/hwang595/ps_pytorch/blob/master/src/run_pytorch.sh). |
| `network`                        | Types of deep neural nets, should be set to the same value as [run_pytorch.sh](https://github.com/hwang595/ps_pytorch/blob/master/src/run_pytorch.sh). |
| `dataset`                  | Datasets use for training, should be set to the same value as [run_pytorch.sh](https://github.com/hwang595/ps_pytorch/blob/master/src/run_pytorch.sh). |
| `model-dir`                       | Directory to save model checkpoints for evaluation, should be set to the same value as [run_pytorch.sh](https://github.com/hwang595/ps_pytorch/blob/master/src/run_pytorch.sh). |

## Future Work
Those are potential directions we are actively working on, stay tuned!
* Explore the use of Atomo with Fourier decompositions, due to its utility and prevalence in signal processing.
* Examine how we can sparsify and compress gradients in a joint fashion to further reduce communication costs.
* Explore jointly sparsification of the SVD and and its singular vectors.
* Integrate ATOMO to state-of-the-art PS (or distributed) frameworks e.g. [Ray](https://rise.cs.berkeley.edu/projects/ray/).

## Citation

```
@inproceedings{wang2018atomo,
  title={ATOMO: Communication-efficient Learning via Atomic Sparsification},
  author={Wang, Hongyi and Sievert, Scott and Liu, Shengchao and Charles, Zachary and Papailiopoulos, Dimitris and Wright, Stephen},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9871--9882},
  year={2018}
}
```
##changelog
###2019-5-21
- 在LSS_SGD_cluster分支上测试cluster方法的效果，详细见该分支日志
###2019-5-22
- 取resnet18在cifar10上的梯度，batchsize 128\*8，每10个step取一次，在cifar100上的梯度，batchsize 128\*8，每200个step取一次
- 安装pytorch0.4.0测试在qsgd上的速度，decode速度还是比pytorch0.3.0慢了10倍，而且在cifar10上SVD方法mpi仍然会报错
- 将梯度画成图，卷积层基本集中分布在0，bn和shotcut参数高斯分布，全连接层的分布比较分散
- pytorch0.3.0imagenet训练需要测试Torchvision的函数是否可用
- 用真实的梯度数据测试，试着解决数据偏斜问题中发现的一些情况：类簇的entropy相似3.**；bucket增大压缩时间减少，cluster数量增大，压缩时间增大，总的bucketsize可以作为一个tradeoff；clustercenter可以作为调节对聚类中心数据大小自适应能力的参数，应该在getarraysize中考虑；真实的梯度中类簇之间的确是数据分布不均匀，但是这种不均匀和梯度的不均匀不成比例，猜测是因为把太大的数值舍弃了的缘故（大数值应该在梯度训练中被考虑，而不是被认为是离群点）
- imagenet数据resnet应该是224，改用官方模型吧
###2019-5-23
- pytorch0.3.0训练imagenet遇到MPI Bcast Truncted错误
- 将每个数据画成图,全连接层数据变化范围较大
- 读sketchML论文
- 分别统计大数值和小数值的误差,大数值的误差比小数值少十倍。
```
big num error 0.20622697472572327, small num error 2.626340389251709
```
###2019-5-24
- 读sketchML论文，思考每个类簇中如何降低每个bucket的误差。目前来看，大数值相对误差还可以控制，但是小数值的相对误差就有些大了，见下表，包含小数值的最大误差都很大；但是根据图片来看，大误差的数据占比很小；而且如何降低positive error，尤其是大数值的positive error，目前来看，正负误差数量差不多；问题的关键在于hash到bucket之后每个桶里的数据是平均的，而且每个类簇的范围似乎还挺大的。
- 下一步可以把采样的数据聚类结果中，每一类的数据的直方图画出来,看看类簇范围
- 在实际做实验的时候应该注意设置聚类采样率，并且观察小数值的误差对梯度下降的影响

| positive_error.max() | np.abs(negative_error).max() |small_score_error.max() | big_score_error.max() |
|---------------------|------------|------------|------------------|
| 115943.33 |90833.9 |115943.33 | 0.7291682 |

- 理论证明部分，需要论证解压后的梯度与随机梯度之间的误差平方的期望有个上界，这个上界小于某个和随机梯度相关的数值
- 深度压缩感知的论文应该仔细读懂 

###2019-5-31
- entropy计算方法
- 二分查找
- 归一化
- 离群点用大的arraysize
 ###2019-6-3
 - 离群点entropy计算，用所有的数据点？
 - 如果不将离群点单独计算，会造成entropy=0，部分bucket里为空，需要修改getlsstable方法才能保证程序正确
 - 数据归一化之后效果并不明显，修改arraysize计算方法，去掉density，但是大数值的arraysize=1
 - 如何设计getarraysize的方法，是的大数值能够分得bucket多一些，而不是仅仅考虑到其entropy很小，density有没有用？
 - 二值网络与signSGD一起训练
 - 自适应的梯度量化方法
 - 压缩感知