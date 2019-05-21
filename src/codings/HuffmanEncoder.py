from dahuffman import HuffmanCodec
class HuffmanEncoder(object):

    def __init__(self):
        self.encoded_index = None
        self.codebook = None
    def encode(self, index):
        self.codebook = HuffmanCodec.from_data(index)
        self.encoded_index = self.codebook.encode(index)
        return self.encoded_index

    def decode(self, encoded_index):
        return self.codebook.decode(encoded_index)