import torch
from LSS import LSS
from svd import SVD
from qsgd import QSGD
from csvec import CSVec
from cmvec import CMVec
import time
import math

import numpy as np

import random
import pickle, sys
from scipy.stats import scoreatpercentile, wasserstein_distance
# from HuffmanEncoder import *

# import jpype
# from jpype import JavaException 
def test_svd(gradients, quantization_level):    
    svd_total = SVD(rank=quantization_level)

    current_time = time.time()
    code = svd_total.encode(gradients)
    encode_time = time.time()-current_time
    

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(pickle_code)
    print('code size {}'.format(code_size))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    decode_value = svd_total.decode(unpickle_code, cuda=False)
    decode_time = time.time() - current_time
    print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.shape))
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)

    print('distance of svd: {}'.format(distance))
    # code_error = (gradients.flat[:] - decode_value.numpy().flat[:])/(np.abs(gradients.flat[:]))
    # big_score = scoreatpercentile(np.abs(gradients.flat[:]), 95)
    # big_mask = np.abs(gradients.flat[:])>big_score
    # small_mask = np.abs(gradients.flat[:])<=big_score
    # big_score_error = np.abs((gradients.flat[:])[big_mask] - (decode_value.numpy().flat[:])[big_mask])/np.abs((gradients.flat[:])[big_mask])
    # # small_score_error to do
    # small_score_error = np.abs((gradients.flat[:])[small_mask] - (decode_value.numpy().flat[:])[small_mask])/np.abs((gradients.flat[:])[small_mask])
    # positive_error = code_error[code_error > 0]
    # negative_error = code_error[code_error <= 0]
    # print(positive_error,positive_error.size, negative_error, negative_error.size)
    # print('mean error{}, positive error {}, negative error {}, big num error {}, small num error {}'.format(code_error.mean(), positive_error.mean(), np.abs(negative_error.mean()), big_score_error.mean(), small_score_error.mean()))
    # print(positive_error.max(), np.abs(negative_error).max(), small_score_error.max(), big_score_error.max())

def test_terngrad(gradients, quantization_level):
    qsgdkw = {'quantization_level':quantization_level, 'scheme' : 'terngrad'}
    qsgd_total = QSGD(**qsgdkw)    
    
    current_time = time.time()
    code = qsgd_total.encode(gradients)
    encode_time = time.time()-current_time

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(pickle_code)
    print('code size {}'.format(code_size))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    decode_value = qsgd_total.decode(unpickle_code, cuda=False)
    decode_time = time.time() - current_time
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)

    print('distance of terngrad: {}'.format(distance))

def test_qsgd(gradients, quantization_level):
    qsgdkw = {'quantization_level':quantization_level}
    qsgd_total = QSGD(**qsgdkw)    
    
    current_time = time.time()
    code = qsgd_total.encode(gradients)
    encode_time = time.time()-current_time

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(pickle_code)
    print('code size {}'.format(code_size))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    decode_value = qsgd_total.decode(unpickle_code, cuda=False)
    decode_time = time.time() - current_time
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)

    print('distance of qsgd: {}'.format(distance))
    # print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.numpy().shape))
    # code_error = (gradients.flat[:] - decode_value.numpy().flat[:])/(np.abs(gradients.flat[:]))
    # big_score = scoreatpercentile(np.abs(gradients.flat[:]), 95)
    # big_mask = np.abs(gradients.flat[:])>big_score
    # small_mask = np.abs(gradients.flat[:])<=big_score
    # big_score_error = np.abs((gradients.flat[:])[big_mask] - (decode_value.numpy().flat[:])[big_mask])/np.abs((gradients.flat[:])[big_mask])
    # # small_score_error to do
    # small_score_error = np.abs((gradients.flat[:])[small_mask] - (decode_value.numpy().flat[:])[small_mask])/np.abs((gradients.flat[:])[small_mask])
    # positive_error = code_error[code_error > 0]
    # negative_error = code_error[code_error <= 0]
    # print(positive_error,positive_error.size, negative_error, negative_error.size)
    # print('mean error{}, positive error {}, negative error {}, big num error {}, small num error {}'.format(code_error.mean(), positive_error.mean(), np.abs(negative_error.mean()), big_score_error.mean(), small_score_error.mean()))
    # print(positive_error.max(), np.abs(negative_error).max(), small_score_error.max(), big_score_error.max())

def test_lss(gradients, quantization_level):
    lss = LSS(bin_num=2000, cluster_num=quantization_level)
 
    
    current_time = time.time()
    compressor = lss.train_cluster(gradients)
    code = lss.encode(gradients, compressor)
    encode_time = time.time()-current_time

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(bytearray(pickle_code))
    print('code size {}, {},{}'.format(code_size, sys.getsizeof(pickle.dumps(code['coded_data'])), sys.getsizeof(pickle.dumps(code['coded_index']))))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()
    grad_number = unpickle_code['flatten_size']
    lss_table = unpickle_code['coded_data']
    # decode huffman code with codebook directly
    codebook = unpickle_code['codebook']
    keysInGroup = codebook.decode(unpickle_code['coded_index'])

    decode_value = lss.decode(lss_table, keysInGroup, grad_number)
    decode_time = time.time() - current_time
    print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.shape))
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)
    print('distance of lss: {}'.format(distance))
    # code_error = (gradients.flat[:] - decode_value.numpy().flat[:])/(np.abs(gradients.flat[:]))
    # big_score = scoreatpercentile(np.abs(gradients.flat[:]), 95)
    # big_mask = np.abs(gradients.flat[:])>big_score
    # small_mask = np.abs(gradients.flat[:])<=big_score
    # big_score_error = np.abs((gradients.flat[:])[big_mask] - (decode_value.numpy().flat[:])[big_mask])/np.abs((gradients.flat[:])[big_mask])
    # # small_score_error to do
    # small_score_error = np.abs((gradients.flat[:])[small_mask] - (decode_value.numpy().flat[:])[small_mask])/np.abs((gradients.flat[:])[small_mask])
    # positive_error = code_error[code_error > 0]
    # negative_error = code_error[code_error <= 0]
    # print(positive_error,positive_error.size, negative_error, negative_error.size)
    # print('mean error{}, positive error {}, negative error {}, big num error {}, small num error {}'.format(code_error.mean(), positive_error.mean(), np.abs(negative_error.mean()), big_score_error.mean(), small_score_error.mean()))
    # print(positive_error.max(), np.abs(negative_error).max(), small_score_error.max(), big_score_error.max())

def test_csvec(gradients):
    vec = torch.tensor(gradients, device='cuda')
    cs_sketch = CSVec(vec.size()[0], c=20000, r=5)
    cs_sketch.accumulateVec(vec)
    sketch_table = cs_sketch.table
 
    
    
    code_size = sketch_table.element_size()*sketch_table.numel()
    print(f'code size {code_size}')


    decode_value = cs_sketch._findAllValues().cpu().numpy()
    print(f'decode shape {decode_value.shape}')
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)
    print(f'distance of cs_sketch: {distance}')


def test_cmvec(gradients):
    vec = torch.tensor(gradients, device='cuda')
    cm_sketch = CMVec(vec.size()[0], c=20000, r=5)
    cm_sketch.accumulateVec(vec)
    sketch_table = cm_sketch.table
 
    
    
    code_size = sketch_table.element_size()*sketch_table.numel()
    print(f'code size {code_size}')


    decode_value = cm_sketch._findAllValues().cpu().numpy()
    print(f'decode shape {decode_value.shape}')
    # distance = wasserstein_distance(gradients, decode_value, np.abs(gradients), np.abs(gradients))
    distance = wasserstein_distance(gradients, decode_value)
    print(f'distance of cm_sketch: {distance}')


if __name__ == '__main__':
    filepath_conv = '../../../gradients_resnet34/conv_layer_csv/gd_conv_data_total.csv'
    filepath_fc = '../../../gradients_resnet34/fc_layer_csv/gd_fc_data_total.csv'
    origin_con_value = np.loadtxt(filepath_conv, delimiter=',', skiprows=1)
    print(origin_con_value.shape, sys.getsizeof(origin_con_value))
    gradients_conv = origin_con_value[:,1:21]
    # print(gradients)
    gradients_conv = gradients_conv.flat[:]*10000

    origin_fc_value = np.loadtxt(filepath_fc, delimiter=',', skiprows=1)
    print(origin_fc_value.shape, sys.getsizeof(origin_fc_value))
    gradients_fc = origin_fc_value[:,1:21]
    # print(gradients)
    gradients_fc = gradients_fc.flat[:]*10000

    gradients = np.concatenate((gradients_fc, gradients_conv))
    # gradients = gradients_conv   
    # jvmPath = jpype.getDefaultJVMPath()           #the path of jvm.dll 
    # classpath = "/Users/keke/Documents/Project/Sketch_DNN/LSS/LSS/sketch/target/classes"                 #the path of PasswordCipher.class 
    # jvmArg = "-Djava.class.path=" + classpath 
    # extpath = "/Users/keke/Documents/Project/Sketch_DNN/LSS/LSS/lib"
    # jvmlib = "-Djava.ext.dirs=" + extpath
    # if not jpype.isJVMStarted():                    #test whether the JVM is started 
    #     jpype.startJVM(jvmPath,jvmArg, jvmlib)             #start JVM 
    # javaClass = jpype.JClass("org.dma.sketchml.sketch.sample.App")   #create the Java class 
    # try: 
    #     java_array = jpype.JArray(jpype.JDouble, 1)(origin_value.tolist())
    #     javaClass.dense()
    # except: 
    #     print ("Unknown Error")
    # finally: 
    #     jpype.shutdownJVM()        #shut down JVM
    print('lss test')
    test_lss(gradients, 2)
    print('qsgd test')
    test_qsgd(gradients, 2)
    print('terngrad test')
    test_terngrad(gradients, 2)
    print('svd test')
    test_svd(gradients, 3)
    print('cs_sketch test')
    test_csvec(gradients)
    print('cm_sketch test')
    test_cmvec(gradients)