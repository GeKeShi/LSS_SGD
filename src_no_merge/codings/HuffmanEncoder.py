from dahuffman import HuffmanCodec
import numpy as np
import pickle,sys
class HuffmanEncoder(object):

    def __init__(self):
        self.encoded_index = None
        self.codebook = None
    def encode(self, index):
        self.codebook = HuffmanCodec.from_data(index)
        self.encoded_index = self.codebook.encode(index)
        return self.codebook, self.encoded_index
    @staticmethod
    def decode(encoded_index, codebook):
        codebook = codebook
        return codebook.decode(encoded_index)


if __name__ == '__main__':
    encoder = HuffmanEncoder()
    coodbook, index = encoder.encode(np.array(range(1000), dtype=int))
    print(sys.getsizeof(pickle.dumps(index)), sys.getsizeof(np.array(range(1000), dtype=int)),index)

