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

 ###2019-6-12
 - 二分法完成，将正负值分开寻找最近的数值，有利于寻找小数值的压缩准确性
 - imagenet程序可以跑通，用Torchvision vgg11_bn以及resnet18,需要去掉测试代码，否则会报错

 ###2019-6-13
 - 编码时间包括了pickle时间和GPU->CPU的时间
 - 多个节点总的上行通信时间应该用master的gather时间来考虑
 - 下行通信时间占用时间很短？

 ###2019-6-14
 - Python2中应struct.pack来二进制化int
 - 每一层的梯度数据量如果太少，会导致有一些bucket里面没有数据，因此允许LSStable里面LSSentry值返回0
 - codebook传输问题解决
 - 压缩后数据量比SVD大三分之一

 ###2019-6-15
 - blosc更换影响？cifar10训练
 - cifar100提前结束？step
 - 安装numpy测试lss

 ###2019-6-18

 ```python
 class HorovodAllreduce(torch.autograd.Function):
    """An autograd function that performs allreduce on a tensor."""

    @staticmethod
    def forward(ctx, tensor, average, name):
        ctx.average = average
        handle = allreduce_async(tensor, average, name)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        return allreduce(grad_output, ctx.average), None, None


def allreduce(tensor, average=True, name=None, compression=Compression.none):
    """
    A function that performs averaging or summation of the input tensor over all the
    Horovod processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Horovod processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    tensor_compressed, ctx = compression.compress(tensor)
    summed_tensor_compressed = HorovodAllreduce.apply(tensor_compressed, average, name)
    return compression.decompress(summed_tensor_compressed, ctx)
```
###6-26
- cifar10/100 res34 batchsize 256/128
- imagenet res50 batchsize 64
- 查找zipml开源实现，bert模型训练梯度
- 比较lss，sketchml，svd，qsgd，TernGrad，zipml等方法resnet18/34/50/vgg16/bert imagenet上固定压缩空间的准确率，取这些网络的梯度，将梯度整体拼接、卷积和全连接层分别拼接，进行压缩。先从原理上进行比较，理论分析各自的优缺点
- cifar10/cifar100上的收敛效果
- 论文框架，算法框架，创新点，哪些实验

###6-27
- lss训练太慢，cifar100需要resnet18 batchsize=512 node=24才能在10天训练到prec5=0.95
- 采集梯度数据
- gn6,gn9,gn10:resnet34 cifar10 svd / cifar100 svd
- gn12,gn21,gn29:resnet34 cifar10 lss / cifar100 lss
- gn2 #0-1: resnet18, 2-3: resnet34 batchsize 128
- gn30 #0-1: resnet50, batchsize 90,  2-3: vgg16 batchsize 24

###6-28
- 将所有卷积层、全连接层的梯度画在同一层，每一层用不同颜色，观察取值变化范围
- 将全部剃度进行压缩，去掉code中的shape，用模型本身的信息取解压
- 将梯度分为卷积层和全连接层两部分进行压缩，用标志位进行标识

###6-29
- 修改为全梯度压缩版本，期望全梯度压缩版本可以有效降低压缩和解压时间
- QSGD论文读懂
- gn6,gn9,gn12:resnet34 cifar10 sgd 256/ cifar100 sgd 128
- gn10,gn21,gn29:resnet34 cifar10 qsgd 256/ cifar100 qsgd 128


###7-1
- 关于大梯度值的优势，提取梯度，比较svd，qsgd，sketchml，lss这些梯度每个bucket中梯度的平均误差和方差的分布，通过比较这些误差来说明在大梯度处lss方法的优势
- 在梯度的压缩效率上，和svd这些方法相比，应用sketchml的minmaxsketch方法
- 在梯度压缩时间上，和sketchml进行比较
- 需要在python上实现sketchml？或者比较这两种算法的算法复杂度
- 单独做一个lss方法的PS架构，1.ps收集10步的梯度，构建聚类模型，将聚类模型同步到所有节点，2.worker收到聚类模型之后，将所有的梯度进行合并后进行压缩，然后将lsstable传输到ps进行merge 3.merge以及reduce之后的lsstable直接分发到所有的计算节点上，计算节点直接在lsstable上查询进行梯度更新 

### 7-2
- 实验现象：svd适合分层，分层的情况下，svd可以有更多的rank，例如合并梯度上rank=2，在3\*3\*512*512的卷积层梯度上，svd可以有18个rank，这也说明svd和pca一样利用了每一层梯度内的线性相关性；
- 分层梯度上，9437328 -> lss 903489 mean error14.22773551940918, positive error 101.96533203125, negative error 73.4845962524414, big num error 0.23505018651485443, small num error 92.32752990722656
  svd 1049102 mean error-35.144351959228516, positive error 270.7290954589844, negative error 338.3236999511719, big num error 1.6632803678512573, small num error 320.6239318847656
  qsgd 1455806  mean error49.60647964477539, positive error 5416.9501953125, negative error 5481.15625, big num error 9.754800796508789, small num error 5734.82177734375
- 在合并的梯度上进行压缩，lss方法更加适合；压缩率最高，44695944/3990709, mean error2.825591802597046, positive error 59.232295989990234, negative error 53.03730773925781, big num error 0.46975284814834595, small num error 59.04869842529297
  svd 22348304, ean error-1.0423961877822876, positive error 4.691502094268799, negative error 6.776299953460693, big num error 0.31322455406188965, small num error 6.019195556640625
  qsgd 6896406  mean error25.608240127563477, positive error 2317.140869140625, negative error 2227.846435546875, big num error 15.815743446350098, small num error 2390.8759765625

- idea，现有压缩方法适合分层压缩，可以进行merge，将网络后面的通信密集型参数先行压缩和通信，同时继续进行反向传播，前面的梯度继续进行计算。可以在目前这个框架基础上设计实验和简单的梯度合并策略



###7-3
- sketchml中有zipml
- APPtest.java文件中double toarray报错
- jpyep错误未知，尝试将apptest的问题解决掉
- server：send step, send parameter, (if step%m=0( 4.receive cluster center&lsstable, 5.train cluster centers&merge lsstable（每一类的几个hash table取最长的，作为merge后的table）, 6.bcast cluster center&lsstable)), gather encoded grad, (10.reduce encoded grad within code（将lsstable和keys取出来，table对应元素取平均，keys对应元素取最大值，得到reduce的lsstable和keys）, 11.用平均之后的lsstable和keys进行decode，decode得到的梯度按照模型每层参数大小存入buffer中等待进行梯度更新), sgd step, step + 1 
- worker：receive step, receive parameter, forward&backward, (1.merge grad, (if step%m=0 (2.train cluster centre, 3.send cluster center & lss table/ bcast cluster center & lss table, 7.receive&update cluster center&lsstable), 8.encode grad), 9.send grad

###7-4
- 下行数据因为是模型参数，所以不能用梯度压缩的方法，即使是下行传输梯度，在sgd中进行查询更新参数的方法，其实和先进行全部查询，再进行更新的效率差不多，因此只需要研究上行数据的压缩效率即可知道下行数据如果也是传递梯度的话，会是什么样的压缩效率。但是下行数据是梯度的话，可以直接传输merge后的聚合梯度，减少了梯度解压和再次压缩的过程
TODO
- 修改encode方法，增加train_cluster(grad)函数返回cluster_model={'centers':centers,'lsstable':lsstable}, encode(grad)函数需要修改为encode(grad, cluster_model),使用聚类模型参数和lsstable来压缩
- 修改decode函数，将解压后的梯度转化为每一层的梯度，存到grad_aggregate_buffer中，修改coder.decode方法,利用传进来的lsstable和keys只进行查询操作
- 3个类别时候，霍夫曼编码的效率
- 稀疏化的实现和效果，lss的稀疏化可以将聚类中心绝对值最小的1个类别认为是0，keysingroup进行稀疏化，这样对于稀疏化之后的数据，其精度仍然可以保持不变，但是需要考虑稀疏化数据keysingroup的聚合问题


### 7-6
- 修改LSS代码。增加traincluster函数，返回compressor，encode方法接收compressor进行压缩， 修改LSSSimplified.py文件，LSSTable函数__init__增加初始化LSSTable
- 修改decode方法，接收lsstable，keysingroup, 在LSSSimplifiedcompressor中需要得到模型参数总共有多少
- gradients_buffer memreset修改形状,修改buffer相关函数
- 不太适用于SSP模型，但是分布式DL中应用的都是BSP模型，尤其是在通信开销比较少，DL是一个非凸问题，参考Chen, Jianmin, Xinghao Pan, Rajat Monga, Samy Bengio, and Rafal Jozefowicz. “Revisiting Distributed Synchronous SGD.” ArXiv:1604.00981 [Cs], April 4, 2016. http://arxiv.org/abs/1604.00981.   Cui, Henggang, Hao Zhang, Gregory R. Ganger, Phillip B. Gibbons, and Eric P. Xing. “GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server.” In Proceedings of the Eleventh European Conference on Computer Systems, 4:1–4:16. EuroSys ’16. New York, NY, USA: ACM, 2016. https://doi.org/10.1145/2901318.2901323.

### 7-8
- 修改clustermodel bcast 为isend
- 解决mpi段错误
- 压缩后的数据量和getsizeof不相符，仅能压缩四倍
- 压缩时间很长的问题

### 7-10
- mpi 报错是因为bucket太大，超过500就报错，因此需要将compressor变成字典
- code_data在linux系统上比mac上的huffman编码大了4倍
- merge key的方法，取最大值的方法不行，变化太大，改为取聚类类别int值的均值，准确的应该是计算相应聚类中心的加权平均，得用cuda来做。因此merge key的具体方法仍然需要讨论
- 用其他worker的clustermodel是可行的，

### 7-11
- keys用bitmap来代替，bitmap可以进行按位或操作，以此进行keys merge

## 2020-12-26
- 调用接口：
```python
    # initialize LSS class
    lss = LSS(cluster_num=quantization_level)   
    current_time = time.time()
    # firstly train a clustering model in sampled data if the data number is greater than 100,000 or training directely. Set encode flag to false if number less than 1,000. Retrun a compressor (object of LSSSimplifiedCompressor)
    compressor = lss.train_cluster(gradients)
    # encode the gradient with the compression, gradient must be a vector, code is a diction: code = {'coded_data':coded_data, 'coded_index':coded_index, 'codebook':codebook, 'encode_flag':encode_flag, 'flatten_size':n}
    code = lss.encode(gradients, compressor)
    encode_time = time.time()-current_time

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(bytearray(pickle_code))
    print('code size {}, {},{}'.format(code_size, sys.getsizeof(pickle.dumps(code['coded_data'])), sys.getsizeof(pickle.dumps(code['coded_index']))))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    # get the number of gradients
    grad_number = unpickle_code['flatten_size']
    # get the sketch
    lss_table = unpickle_code['coded_data']
    # get the mapping relation between gradient index and cluster index, decode with huffman decoder, which is a static variable in class LSSSimplifiedCompressor
    # actually, we can use the codebook to decode directely: codebook.decode(encoded_index)
    keysInGroup = codings.LSSSimplifiedCompressor.denseEncoders.decode(unpickle_code['coded_index'], unpickle_code['codebook'])
    ...

def _decode_total_grad(self, coded_msgs):
  # decode with lss object (self._coder)
  grad = self._coder.decode(self.aggregate_total_gradient(coded_msgs))
```
- how to test lss/qsgd/terngrad: level=1bit: set level=2, set `pctthreshold=99.9` in `clusterNoThreshold` ; other level: set `ClusterArraychoicemethod=3` in `LSS.py` to manually set bucket number, then set the bucket number in `LSSSimlplified.py:255` according to the `pctThreshold=xx`in and number of gradients, usually 10% of the total number of gradients for the biggest cluster.
- conv*21+fc*21=1058560; conv*21=34560

| quant_type| 1bit  |     | 2 bit ||  3bit | |
|------|------|-------|-------|------|-------|-------|
| gradients|conv*m |conv*m+fc*n|conv*m|conv*m+fc*n|conv*m|conv*m+fc*n|
|cas| 32.14046246745134|11.147359768733848 |11.282536098382078  |6.799291919266389| 7.152148586139356| 4.6468552070535125|
|qsgd| 1201.346920481589 | 542.9087644726227|342.60097550533305  |160.52569309063284| 100.97366302405044| 54.22812858660813|
|terngrad| 121.51607863103308| 128.6203687568705| 19.781274792370304  |29.129952586313525| 4.813074015979637|8.448870630410452 |
|cs_sketch| 321.0532244372209| 148.09765508413216| 198.38154942443302 |90.5277719795757| 97.19696268444254| 82.72278706892754|
|cm_sketch| 871.80470766876 | 147.42656203177995|351.84080446812715 |93.14706263411549| 125.12181149932938| 83.03590199229424|

| quant_type| 1bit  |     | 2 bit ||  3bit | |
|------|------|-------|-------|------|-------|-------|
| gradients|conv*m |conv*m+fc*n|conv*m|conv*m+fc*n|conv*m|conv*m+fc*n|
|cas| 100| 3000|11.282536098382078  |6.799291919266389|3116+1579+408+357+1505+3036| 9000+ 4000+ 10+ 10+ 4000+ 9000|
|cs_sketch| 216*5| 6000*5| 432*5 |13232*5| 648*5 | 19848*5 |

- test merge efficiency with 4-512 nodes, 1000,000 gradients

test merge for 4 workers
merge time 0.0006134510040283203 seconds, decode time 0.0021097660064697266 seconds
test merge for 8 workers
merge time 0.0007631778717041016 seconds, decode time 0.004918813705444336 seconds
test merge for 16 workers
merge time 0.0011944770812988281 seconds, decode time 0.004057884216308594 seconds
test merge for 32 workers
merge time 0.0020890235900878906 seconds, decode time 0.007892847061157227 seconds
test merge for 64 workers
merge time 0.0037355422973632812 seconds, decode time 0.1314220428466797 seconds
test merge for 128 workers
merge time 0.007250308990478516 seconds, decode time 0.5949215888977051 seconds
test merge for 256 workers
merge time 0.014047622680664062 seconds, decode time 1.515331506729126 seconds
test merge for 512 workers
merge time 0.027673721313476562 seconds, decode time 3.354495048522949 seconds

1.4
- 3320 gn4 sgd c10 res34
- nohup sh run_pytorch.sh >> 104-ren34-c10-qsgd-lr01.log 2>&1 & level1 gn6 3537 end
- nohup sh run_pytorch.sh >> 104-res34-c10-sgd-lr01.log 2>&1 & [2] 2916 gn6 end
- nohup sh run_pytorch.sh >> 104-res34-c10-terngrad-level1-lr01.log 2>&1 & gn4 3769  end
- nohup sh run_pytorch.sh >> 104-res34-c10-qsgd-level1-lr01.log 2>&1 & gn6 3118 end

1.5
- nohup sh run_pytorch.sh >> 104-res34-c10-terngrad-lr005.log 2>&1 & gn4 20130 end
- nohup sh run_pytorch.sh >> 105-res34-c10-qsgd-lr005.log 2>&1 & gn4 20306 end
- nohup sh run_pytorch.sh >> 106-Res34-c10-lss-level4-lr065-256.log 2>&1 & from gn6 6399 end

       JOBID PARTITION                                     NAME     USER ST         TIME  NODES NODELIST(REASON)
     2798039    TH_GPU                              sleeping.sh    wql17  R         1:09      1 gn1
     2798040    TH_GPU                              sleeping.sh    wql17  R         1:09      1 gn2
     2798041    TH_GPU                              sleeping.sh    wql17  R         1:09      1 gn13
     2798042    TH_GPU                              sleeping.sh    wql17  R         1:09      1 gn14

- nohup sh run_pytorch.sh >> 105-Res50-img-sgd-lr01-128.log 2>&1 & gn13,14 from gn1:18100
- 

1.9 attention56 (55703780,) 222815216 lss huffman vs. terngrad
4bit:clusterdensity [0.1500077  0.0884799  0.20087017 0.56064223 0.57630681 0.19383636
3bit:clusterdensity [0.15001444 0.16593206 0.68405351 0.68485795 0.16513685 0.1500052 ]
2bit:clusterdensity [0.15001638 0.84998362 0.84999688 0.15000312]
 0.07984435 0.15001249]
|level|2bit|3bit|4bit|
|----|------|----|--------|
|lss| 13344694|16301318 | 18680972|
|qsgd| 36017287|45592710 | 53427165|

1.10
- nohup sh run_pytorch.sh >> 110-Res34-c100-lss-level4-lr065-256.log 2>&1 & gn6 25720 [1] 25707 end level 3
- nohup sh run_pytorch.sh >> 110-Res34-c100-qsgd-level1-lr005-128.log 2>&1 & gn6 25941 [2] 25929

1.12
- nohup sh run_pytorch.sh >> 112-res34-c100-sgd-lr065-256.log 2>&1 & gn4[1] 28796 28806
- nohup sh run_pytorch.sh >> 112-res34-c100-lss-lr065-256.log 2>&1 & gn4 29035[2] 29023 level2
- nohup sh run_pytorch.sh >> 112-res34-c100-lss-level1-lr04-256.log 2>&1 &gn6[3] 29171

relative_error  wasserstin_distance  [90000,30000,10000,1][80000,1][40000]

| quant_type| 1bit  |     | 2 bit ||  3bit | |
|------|------|-------|-------|------|-------|-------|
| gradients|res50|attention|res50|attention|res50|attention|
|cas| 0.45703603594737624 0.00018928746217576706|0.5515058914458363 0.00017467930261935624 |0.6629707191839835 0.0001229379612465994 |0.3498290280226604 0.00010187067650051319| 0.6117118185989174 8.371413342497276e-05| 0.2572852323261022 5.8881512359573626e-05|
|qsgd| 465.010196295329 0.006396862295065109 | 462.9645509349531 0.005488469182881338|449.11593486528784 0.006176206009391841  |447.5063900171538 0.005290591524172797| 43.13614901352159 0.0018414205499548114| 42.54519917821269 0.001567113519621464|
|terngrad| 11.904052001056868 0.0007922541716669365| 8.27560963662824 0.0005649908533541372| 10.796809562046027 0.0006300023624475679 |7.325388132456993 0.00042835059752627377| 1.0976560577688608 0.00013523636736958972|0.7938491755840088 9.191594490853879e-05 |
|cs_sketch| 28.177256948120696 0.0020659879121065297| 42.54450650351792 0.001553707124264352| 12.37576698547023 0.001361186417215902 |20.583004984658302 0.0010342708275154935| 8.25204325638012 0.001073722214972378| 13.202235040259783 0.0007988661824981198|
|cm_sketch| 31.492239047865006 0.0022094028701950437 | 43.190212561166845 0.0015701503974317652|13.286378103992869 0.001426836894922324 |20.74610619718759 0.0010397253765544936| 8.688759392772553 0.0011117196119936402| 13.28564379355246 0.0008023498193210307|