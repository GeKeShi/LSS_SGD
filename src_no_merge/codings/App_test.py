import random
import numpy as np
import ClusterStaticNoThreshold
import LSSSimplifiedCompressor
import encodeChoicer
class AppTest(object):
    def __init__(self, *args):
        self._random = random.random()
        self.values = args[0]
        self.bin_num = args[1]
        self.cluster_Number = args[2]
        if self.values != None:
            denseLSSCKV2()
    
    def denseLSSCKV2(self):
        #  int n = 1000000;
        # // double density = 0.9;
        # //double[] values = new double[n];

        n = np.size(self.values)
        # ? int traceCount = Math.min((int) Math.round(n), 100000);
        traceCount = min(int(round(n)), 100000)
        traces0 = np.zeros(traceCount)
        sampled = 0

        for i in range(n):
            if sampled < traceCount and ClusterStaticNoThreshold.WhetherAdd2ClusterTrace(self._random, n, traceCount):
                traces0[sampled] = values[i]
                sampled += 1
            

        traces = traces0[0:sampled - 1]
        # traces = self.values
        

        # //Quantizer.QuantizationType quantType = Quantizer.QuantizationType.QUANTILE;
        binNum = self.bin_num
        # //Quantizer.DEFAULT_BIN_NUM;
        _expectedNumItems = n
        
        _clusterCount = LSSSimplifiedCompressor.clusterNumber

        
        # //key encoding
        encodeChoicer = encodeChoice.Huffman

        # //cluster choice
        clusterArrayChoiceMethod = 1
        # to do
        compressor = LSSSimplifiedCompressor(encodeChoicer, _expectedNumItems, _clusterCount, binNum, clusterArrayChoiceMethod,
                traces)

        traces = None
        # //compressor.compressDense(values);
        # long t1 = System.currentTimeMillis();
        compressor.parallelCompressDense(self.values)
        dValues = compressor.decompressDense()
        # time = System.currentTimeMillis() - t1;
        # LOG.info("DecodeAll:" + n + ", TotalDelay: " + time);
        # //LOG.info("LSS:\nFirst 10 values before: " + Arrays.toString(Arrays.copyOf(values, 10)));
        # // LOG.info("First 10 values after:  " + Arrays.toString(Arrays.copyOf(dValues, 10)));

        # //statistics
        # double rmse = 0.0;
        # double re = 0, denominator = 0;
        for i in range(n):
            denominator = 1
            if denominator == 0:
                continue
            else:
                re = abs(dValues[i] - values[i]) / denominator
                # qSketch.update(re);
                rmse += (re) * (re)
