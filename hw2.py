import numpy as np
import copy
import math

# 1.compute exp
def exp(a,n):
    if n == 0:
        return 1
    if n == 1:
        return a
    else:
        if n%2 == 0:
            b = exp(a,n/2)
            return b*b
        else:
            b = exp(a,(n-1)/2)
            return b*b*a


# 2.max contiguous subsequence sum
def MS(ls,p,q,r):
    sum1 = 0; max1 = 0
    sum2 = 0; max2 = 0
    k1 = 0; k2 = 0
    p1 = copy.deepcopy(q)
    p2 = copy.deepcopy(q) + 1
    while p1>=p:
        sum1 += ls[p1]
        if sum1 > max1:
            k1 = p1
            max1 = sum1
        p1 -= 1
    while p2<=r:
        sum2 += ls[p2]
        if sum2 > max2:
            max2 = sum2
            k2 = p2
        p2 += 1
    midsum = max1 + max2
    return midsum,k1,k2


def MCS(ls,p,r):
    if p == r:
        return ls[p],p,r
    q = int((p + r)/2)
    lsum,ll,lh = MCS(ls,p,q)
    rsum,rl,rh = MCS(ls,q+1,r)
    midsum,ml,mh = MS(ls,p,q,r)
    maxsum = max(lsum,rsum,midsum)
    if maxsum == lsum:
        return maxsum,ll,lh
    elif maxsum == rsum:
        return maxsum,rl,rh
    else:
        return maxsum,ml,mh


# 3.integer multiplication
def Im(x,y):
    n = len(x)
    if n == 1:
        return x[0]*y[0]
    a = x[0:int(n/2)]; b = x[int(n/2):]
    c = y[0:int(n/2)]; d = y[int(n/2):]
    pa = Im(b,d)
    pb = Im(a,c)
    pc = Im(a+b,c+d) - pa - pb
    return pb + pc*10**(n/2) + pa*10**n


def IM(x,y):
    l1 = len(x)
    l2 = len(y)
    if l1 >= l2:
        k = 1
        while k < l1:
            k *= 2
        pd1 = k - l1
        pd2 = k - l2
        for i in range(pd1):
            x = np.append(x,[0])
        for i in range(pd2):
            y = np.append(y, [0])
    else:
        k = 1
        while k < l2:
            k *= 2
        pd1 = k - l1
        pd2 = k - l2
        for i in range(pd1):
            x = np.append(x,[0])
        for i in range(pd2):
            y = np.append(y, [0])
    return Im(x,y)


# 4.select
def Partition(ls,p,r,x):
    i = p - 1
    for j in range(p,r+1):
        if ls[j] == x:
            ls[j],ls[r] = ls[r],ls[j]
    for j in range(p,r):
        if ls[j] <= x:
            i += 1
            ls[i], ls[j] = ls[j], ls[i]
    ls[i+1],ls[r] = ls[r],ls[i+1]
    return i+1


def Divide(ls,p,r,group):
    n = math.ceil((r - p + 1)/group)
    m = []
    pt = copy.deepcopy(p)
    for i in range(n):
        if pt+5 <= r+1:
            m.append(np.median(ls[pt:pt+5]))
            pt += 5
        else:
            count = r - pt + 1
            l = ls[pt:r+1]
            for j in range(count-1):
                mini = j
                for k in range(j+1,count):
                    if l[k] <= l[mini]:
                        mini = k
                l[k],l[j] = l[j], l[k]
            m.append(l[int(len(l)/2)])
    return m


def Select(ls,p,r,i):
    if p == r:
        return ls[p]
    m = Divide(ls,p,r,5)
    x = Select(m,0,len(m)-1,int((len(m)-1)/2))
    q = Partition(ls,p,r,x)
    if q == i:
        return ls[q]
    elif i < q:
        return Select(ls,p,q-1,i)
    else:
        return Select(ls,q+1,r,i)


# 5.max profit
def MidProfit(ls,p,q,r):
    minprice = float('inf')
    maxprice = 0
    h = 0; l = 0
    for i in range(p,q+1):
        if ls[i] < minprice:
            minprice = ls[i]
            l = i
    for i in range(q+1,r+1):
        if ls[i] > maxprice:
            maxprice = ls[i]
            h = i
    return (ls[h]-ls[l]),l,h


def MaxProfit(ls,p,r):
    if p == r:
        return 0,p,r
    q = int((p+r)/2)
    lp,ll,lh = MaxProfit(ls,p,q)
    rp,rl,rh = MaxProfit(ls,q+1,r)
    mp,ml,mh = MidProfit(ls,p,q,r)
    maxprofit = max(lp,rp,mp)
    if maxprofit == lp:
        return maxprofit,ll,lh
    elif maxprofit == rp:
        return maxprofit,rl,rh
    else:
        return maxprofit,ml,mh


# 6.find the closest numbers
def ClosestNum():
    pass


if __name__ == '__main__':
    ls = np.random.randint(-10,10,size=10)
    print(ls)
    for i in range(len(ls)):
        print(Select(ls,0,len(ls)-1,i))
