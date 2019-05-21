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
    test = Test()