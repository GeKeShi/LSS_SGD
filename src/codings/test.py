import numpy as np
import os


def groupInputKV(flowValue, clusterCenters):
    line_index = -1
    vat = float("inf")
    for i in range(clusterCenters.size):
        v = abs(clusterCenters[i] - flowValue)
        # //closer and same symbol
        if clusterCenters[i] * flowValue >= 0:
            if v < vat:
                vat = v
                line_index = i
    print(line_index)
    # return index
    
    #binsearch
    rescenters = clusterCenters - flowValue
    if rescenters[0] > 0:
        return 0
    if rescenters[-1] < 0:
        return clusterCenters.size -1

    low = 0
    high =clusterCenters.size -1
    while low < high:
        mid = int((low + high)/2)
        if rescenters[mid]<0:
            if rescenters[mid+1]<0:
                low = mid + 1
            else:
                index = mid if -(rescenters[mid]) < rescenters[mid+1] else mid+1
                if line_index != index:
                    print('False')
                return index
        elif rescenters[mid]>0:
            if rescenters[mid-1]>0:
                high = mid -1
            else:
                index = mid if rescenters[mid] < -(rescenters[mid-1]) else mid-1
                if line_index != index:
                    print('False')
                return index
        else:
            if line_index != mid:
                print('False')
            return mid

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
    a = np.array([-2.05388354e-03,-1.99129572e-03,-1.86800794e-03,-1.74586521e-03
,-1.62792648e-03,-1.52621791e-03,-1.42954313e-03,-1.34222349e-03
,-1.26048131e-03,-1.18082808e-03,-1.09722128e-03,-1.01076509e-03
,-9.31580318e-04,-8.55890219e-04,-7.84054748e-04,-7.15427333e-04
,-6.49811176e-04,-5.85163361e-04,-5.21740410e-04,-4.61032614e-04
,-4.01782920e-04,-3.45435168e-04,-2.91485601e-04,-2.40328911e-04
,-1.92993364e-04,-1.50288019e-04,-1.10734574e-04,-7.40558025e-05
,-3.86986358e-05,-8.82707536e-06, 9.32055991e-06,3.98661941e-05,
7.49204191e-05, 1.13542046e-04,1.53903966e-04,1.99793023e-04
,2.46554438e-04,2.96744343e-04,3.47292225e-04,4.01270372e-04
,4.58342634e-04,5.19697205e-04,5.84859110e-04,6.53620576e-04
,7.25848600e-04,7.96803040e-04,8.71983531e-04,9.45946435e-04
,1.02082768e-03,1.10315403e-03,1.18662091e-03,1.27827143e-03
,1.37516356e-03,1.47426967e-03,1.57419289e-03,1.68085925e-03
,1.78216584e-03,1.89116539e-03,2.00607697e-03,2.06878897e-03])
    a = np.sort(a)
    b = [-3.6676647e-07,-6.979735e-07,-1.3416866e-08,-1.6588092e-07,-3.859766e-07,-3.6155916e-07,-2.2599238e-07,-3.2089588e-07,-2.338594e-07,-1.5242999e-07,-1.16153615e-07,-2.1492633e-07,-3.3966592e-07,-3.236712e-07,-6.860155e-07]
    for item in b:
        print(groupInputKV(item, a))