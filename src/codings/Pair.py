
class Pair(object):  
# //index：类别内元素所占比例；value：中心点；entropy：类别发散程度
    def  __init__(self, index, value, entropyV):
        self.index = index
        self.value = value
        self.entropyVal = entropyV
