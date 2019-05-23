import numpy as np
import os

class Test(object):
    b=5
    def __init__(self):
        self.fun(1)

    def fun(self, a):
        self.b = 6
        print(self.b)
        # Test.b = 7
        print (Test.b,b)
        print(a)

if __name__ == '__main__':
    a = np.zeros(100)
    path = os.path.join('100_'+"a", 'aa')
    print(path)
    os.makedirs(path)
    np.save(os.path.join(path, 'aaa'), a)