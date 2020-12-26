from LSS import LSS
from svd import SVD
from qsgd import QSGD
import time
import math

import numpy as np

import random
import pickle, sys
from scipy.stats import scoreatpercentile, wasserstein_distance
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
    print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.numpy().shape))
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
    lss = LSS(cluster_num=quantization_level)
 
    
    current_time = time.time()
    compressor = lss.train_cluster(gradients)
    code = lss.encode(gradients, compressor)
    encode_time = time.time()-current_time

    pickle_code= pickle.dumps(code)
    code_size = sys.getsizeof(bytearray(pickle_code))
    print('code size {}, {},{}'.format(code_size, sys.getsizeof(pickle.dumps(code['coded_data'])), sys.getsizeof(pickle.dumps(code['coded_index']))))
    unpickle_code = pickle.loads(pickle_code)
    current_time = time.time()

    decode_value = lss.decode(unpickle_code, cuda=False)
    decode_time = time.time() - current_time
    print('encode time {}, decode_time {}, decode shape {}'.format(encode_time, decode_time, decode_value.numpy().shape))
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

if __name__ == '__main__':
    filepath = '/Users/keke/Documents/Project/Sketch_DNN/gradients_resnet34/conv_layer_csv/gd_conv_data_total.csv'
    origin_value = np.loadtext(filepath, delimiter=',', skiprows=1)
    print(origin_value.shape, sys.getsizeof(origin_value))
    gradients = origin_value[:,0]
    gradients = np.flatten(gradients)
    
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
    test_svd(gradients, 4)