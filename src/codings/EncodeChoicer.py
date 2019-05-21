from enum import Enum, unique
@unique
class EncodeChoicer(Enum):
    deltaAdaptive = 0
    DeltaBinary = 1
    Huffman = 2

