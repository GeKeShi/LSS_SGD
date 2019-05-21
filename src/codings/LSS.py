import torch
import time
import math
from coding import Coding
import numpy as np
from ClusterStaticNoThreshold import *
from LSSSimplifiedCompressor import *
from EncodeChoicer import *
import random
import pickle, sys, dill

class LSS(Coding):
    def __init__(self, scheme='lss', bin_num=100, cluster_num=50, *args, **kwargs):
        self.scheme = scheme
        self._random = random.random()
        self.values = None
        self.bin_num = bin_num
        self.cluster_Number = cluster_num
        self.compressor = None

    def encode(self, v, **kwargs):
        if isinstance(v, (torch.Tensor)):
            self.values = v.cpu().numpy().flat[:]
        elif isinstance(v, np.ndarray):
            self.values = v.flat[:]
        else:
            raise ValueError("Object passed to encode not ndarray or torch.Tensor")
        
        n = np.size(self.values)
        shape = np.shape(v)
        # to do 
        if n > 1000:
            encode_flag = True
            # ? int traceCount = Math.min((int) Math.round(n), 100000);
            if n < 100000:
                traces = self.values
            else:
                traceCount = 100000
                # traces0 = np.zeros(traceCount)
                traces0 = []
                # sampled = 0

                # for i in range(n):
                #     if ClusterStaticNoThreshold.WhetherAdd2ClusterTrace(random.random(), n, traceCount):
                #         traces0.append(self.values[i])
                #         sampled += 1
                for i in range(traceCount):
                    traces0.append(self.values[random.randint(0, n-1)])

                # traces = traces0[0:sampled - 1]
                traces = np.array(traces0)
            print(traces.shape)
            # traces = self.values
            

            # //Quantizer.QuantizationType quantType = Quantizer.QuantizationType.QUANTILE;
            binNum = self.bin_num
            # //Quantizer.DEFAULT_BIN_NUM;
            _expectedNumItems = n
            
            # _clusterCount = LSSSimplifiedCompressor.clusterNumber
            _clusterCount = self.cluster_Number
            
            # //key encoding
            # encodeChoicer = EncodeChoicer.Huffman

            # //cluster choice
            clusterArrayChoiceMethod = 1
            # to do
            self.compressor = LSSSimplifiedCompressor(_expectedNumItems, _clusterCount, binNum, clusterArrayChoiceMethod, traces)

            traces = None
            # //compressor.compressDense(values);
            # long t1 = System.currentTimeMillis();
            self.compressor.parallelcompressDense(self.values)
            # coded_data = self.compressor.lssCKInstance.LSSTable
            coded_data = self.compressor.LSSTable_val
            coded_index = self.compressor.encoded_index
            # hash_list = self.compressor.lssCKInstance.LongHashFunction4PosHash
            code = {'coded_data':coded_data, 'coded_index':coded_index, 'encode_flag':encode_flag, 'shape': shape, 'flatten_size':n}
            print('size of coded data {}, size of encoded index {}'.format(sys.getsizeof(code['coded_data']), sys.getsizeof(code['coded_index'])))
            # code = {'coded_data':coded_data, 'coded_index':coded_index,
            #         'shape': shape}
        else:
            encode_flag = False
            code = {'coded_data':v, 'encode_flag':encode_flag}
        if kwargs.pop('timings', False):
            data = {}
            return code, data
        return code

    def decode(self, code, cuda=True, **kwargs):
        """
        Decode the coding.
        ## NumPy
         'comm_wait': 0.0728750228881836,
         'decode_time': 0.1349341869354248,
         'example_to_gpu': 0.0006515979766845703,
         'grad_compute_time': 0.5815503597259521,
         'grad_forward_pass': 0.23496603965759277,
         'grad_variance_increase': 31.754316389320049,
         'iallgather_prepare_time': 0.017401456832885742,
         'isend_time': 0.029105424880981445,
        ## PT GPU
        """

        dValues = LSSSimplifiedCompressor.decompressDense(code)
        # time = System.currentTimeMillis() - t1;
        # LOG.info("DecodeAll:" + n + ", TotalDelay: " + time);
        # //LOG.info("LSS:\nFirst 10 values before: " + Arrays.toString(Arrays.copyOf(values, 10)));
        # // LOG.info("First 10 values after:  " + Arrays.toString(Arrays.copyOf(dValues, 10)));

        # //statistics
        # double rmse = 0.0;
        # double re = 0, denominator = 0;
        # for i in range(n):
        #     denominator = 1
        #     if denominator == 0:
        #         continue
        #     else:
        #         re = abs(dValues[i] - values[i]) / denominator
        #         # qSketch.update(re);
        #         rmse += (re) * (re)
        dValues = torch.Tensor(dValues)
        if cuda:
            dValues = dValues.cuda()
        return dValues

if __name__ == '__main__':
    filepath = './test_grad/conv8_weights_0.npy'
    origin_value = np.load(filepath)
    print(origin_value.shape)
    lss = LSS()
    current_time = time.time()
    code = lss.encode(origin_value)
    encode_time = time.time()-current_time
    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(pickle_code)
    print('code size {}, {},{}'.format(code_size, sys.getsizeof(pickle.dumps(code['coded_data'])), sys.getsizeof(pickle.dumps(code['coded_index']))))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    decode_value = lss.decode(unpickle_code, cuda=False)
    decode_time = time.time() - current_time
    print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.numpy().shape))
    code_error = np.abs(origin_value - decode_value.numpy())/(np.abs(origin_value))
    print(code_error)
    print('mean error{}'.format(code_error.mean()))
