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

##Change log
2019-5-21
- 实验发现arraysize设置有问题，这里本来各个类簇都是相似的，但是总的bucket设置太小，导致arraysize一开始总是10，后面总的bucket余额不足，导致后面的arraysize一直是1，解决方案一个是减少聚类类别数目，另一个是增大总的bucket数量
```
getarraysize 0.01933189099120598
arraysize 10
finalleft 90
getarraysize 0.019398610942923687
arraysize 10
finalleft 80
getarraysize 0.018548518332736963
arraysize 10
finalleft 70
getarraysize 0.018891239233391355
arraysize 10
finalleft 60
getarraysize 0.020302156241378762
arraysize 10
finalleft 50
getarraysize 0.02094761227269493
arraysize 10
finalleft 40
getarraysize 0.020687377105392875
arraysize 10
finalleft 30
getarraysize 0.019054388131587937
arraysize 10
finalleft 20
getarraysize 0.017682702380910697
arraysize 10
finalleft 10
getarraysize 0.017507211678353382
arraysize 10
finalleft 0
getarraysize 0.01835079444332231
arraysize 10
finalleft -1
getarraysize 0.01606530510767647
arraysize 10
finalleft -2
getarraysize 0.016039484803934783
arraysize 10
finalleft -3
getarraysize 0.014236326435342615
arraysize 10
finalleft -4
getarraysize 0.013081756047883593
arraysize 10
finalleft -5
getarraysize 0.013529083779155618
arraysize 10
finalleft -6
getarraysize 0.011692224622733779
arraysize 10
finalleft -7
getarraysize 0.012055544387550781
arraysize 10
finalleft -8
getarraysize 0.01097382115790478
arraysize 10
finalleft -9
getarraysize 0.010453414481855019
arraysize 10
finalleft -10
getarraysize 0.011672702412759963
arraysize 10
finalleft -11
getarraysize 0.011368718560440225
arraysize 10
finalleft -12
getarraysize 0.010753186817084653
arraysize 10
finalleft -13
getarraysize 0.010748615190562949
arraysize 10
finalleft -14
getarraysize 0.009573373855043601
arraysize 10
finalleft -15
getarraysize 0.009474927393438868
arraysize 10
finalleft -16
getarraysize 0.009572922676061184
arraysize 10
finalleft -17
getarraysize 0.009188197247925505
arraysize 10
finalleft -18
```
- getarraysize函数的数值结果如下，量纲不同是否有影响？
```
entropy 3.321488697662998, center 3.31592200382147e-05, density 0.12604, cumsum 1.3881790316094779e-05
entropy 3.3213713619899417, center 2.7699314159690402e-05, density 0.12712, cumsum 2.557679329833868e-05
entropy 3.320994291986186, center 1.963592330866959e-05, density 0.12414, cumsum 3.3672060672850626e-05
entropy 3.2760462266375003, center 1.1846359484479763e-05, density 0.10272, cumsum 3.7658543883607535e-05
entropy 2.4062201532081726, center 3.971729711338412e-06, density 0.02527, cumsum 3.7900045636610606e-05
entropy 2.077551624464981, center 3.884974830725696e-06, density 0.024739999999999998, cumsum 3.8099728009574746e-05
entropy 3.2698014925205796, center 1.1757224456232507e-05, density 0.09522, cumsum 4.176034570050563e-05
entropy 3.3206511715467912, center 1.9784343749051914e-05, density 0.12211999999999999, cumsum 4.978325164734145e-05
entropy 3.3211784334154975, center 2.800445508910343e-05, density 0.12603999999999999, cumsum 6.150595378649958e-05
entropy 3.3213653741889355, center 3.3463194995420054e-05, density 0.12658999999999998, cumsum 7.557560909293602e-05
```
- 增加insert_cluster_label函数返回聚类标签
- 用聚类中心list代替lsstable_val，不再调用get_lsstable
- 解压查询时直接查询聚类类别index取聚类中心作为解压值
- 修改聚类中心数目
- 实验效果表明，insert中聚类的时间占用绝大部分，由于查找相近聚类类别是一个两重循环，因此时间很长，具体为：

|类别数目    |压缩时间/s|
| ----------------------------- | ---------------------------------------- |
|50         |63      |
|20         |25       |
|5          |7.86     |

- 修改聚类样本采样方法，防止大规模数据中正负数极不均衡
- 可以看出仅仅使用聚类方法并不能从根本上改变压缩时间很长的问题，即使使用折半查找也不能从根本上改善，需要结合CUDA多线程来解决
- 聚类技术的用处仅剩下和哈希方法的精度的比较了
- 下一步要做的包括：取真实的训练数据样本来测试精度；将压缩方法融合到训练过程中；CUDA并行插入和查询


