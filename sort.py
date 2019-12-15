import numpy as np
import copy
import math
import time


def InsertSort(ls):
    lt = copy.deepcopy(ls)
    n = len(lt)
    lt.append(0)
    for i in range(1,n):
        lt[n] = lt[i]
        j = i - 1
        while (lt[n] < lt[j]) & (j>=0):
            lt[j+1] = lt[j]
            j -= 1
        lt[j+1] = lt[n]
    lt.pop(n)
    return lt


def BInsertSort(ls):
    lt = copy.deepcopy(ls)
    n = len(lt)
    lt.append(0)
    for i in range(1,n):
        lt[n] = lt[i]
        low = 0; high = i-1
        while low <= high:
            mid = int((low+high)/2)
            if lt[n] <= lt[mid]:
                high = mid - 1
            else:
                low = mid + 1
        j = i - 1
        while j >= low:
            lt[j+1] = lt[j]
            j -= 1
        lt[low] = lt[n]
    lt.pop(n)
    return lt


def ShellSort(ls):
    lt = copy.deepcopy(ls)
    n = len(lt)
    lt.append(0)
    dk = [3,2,1]
    for k in range(len(dk)):
        for i in range(dk[k],n):
            lt[n] = lt[i]
            j = i - dk[k]
            while (lt[n] < lt[j]) & (j>=0):
                lt[j+dk[k]] = lt[j]
                j -= dk[k]
            lt[j+dk[k]] = lt[n]
    lt.pop(n)
    return lt


def BubbleSort(ls):
    lt = copy.deepcopy(ls)
    n = len(lt)
    flag = 1
    for i in range(1,n):
        if flag == 0:
            break
        flag = 0
        for j in range(n-i):
            if lt[j] > lt[j+1]:
                flag = 1
                lt[j], lt[j+1] = lt[j+1], lt[j]
    return lt


def SelectSort(ls):
    lt = copy.deepcopy(ls)
    n = len(lt)
    for i in range(n-1):
        k = i
        for j in range(i+1,n):
            if lt[j] < lt[k]:
                k = j
        lt[i], lt[k] = lt[k], lt[i]
    return lt


def Partition1(ls,low,high):
    pivot = ls[low]
    while low < high:
        while (low<high)&(ls[high]>=pivot):
            high -= 1
        ls[low] = ls[high]
        while (low<high)&(ls[low]<=pivot):
            low += 1
        ls[high] = ls[low]
    ls[low] = pivot
    return low


def Partition2(lt,p,r):
    pivot = lt[r]
    i = p - 1
    for j in range(p,r):
        if lt[j] <= pivot:
            i += 1
            lt[j],lt[i] = lt[i], lt[j]
    lt[i+1],lt[r] = lt[r],lt[i+1]
    return i+1


def QuickSort(ls,low,high):
    if low >= high:
        return ls
    else:
        pivotLoc = Partition2(ls,low,high)
        QuickSort(ls,low,pivotLoc-1)
        QuickSort(ls,pivotLoc+1,high)


def Merge(lt,p,mid,r):
    p1 = copy.deepcopy(p)
    p2 = copy.deepcopy(mid) + 1
    ml = []
    while(p1<=mid)&(p2<=r):
        if lt[p1] <= lt[p2]:
            ml.append(lt[p1])
            p1 += 1
        else:
            ml.append(lt[p2])
            p2 += 1
    if p1 == mid + 1:
        ml.extend(lt[p2:r+1])
    else:
        ml.extend(lt[p1:mid+1])
    k = 0
    for i in range(p,r+1):
        lt[i] = ml[k]
        k += 1
    return None


def MergeSort(lt,p,r):
    if p >= r:
        return None
    else:
        mid = int((p + r)/2)
        MergeSort(lt,p,mid)
        MergeSort(lt,mid+1,r)
        Merge(lt,p,mid,r)


def Partition3(lt, p, r, x):
    i = p - 1
    for j in range(p, r + 1):
        if lt[j] <= x:
            i += 1
            lt[i], lt[j] = lt[j], lt[i]
    return i+1


def Divide(lt,p,r):
    n = math.ceil((r - p + 1)/5)
    l = []
    m = []
    pt = copy.deepcopy(p)
    for i in range(n):
        if pt+5 <= r:
            l.append(lt[pt:pt+5])
            pt += 5
        else:
            l.append(lt[pt:r+1])
        m.append(np.median(np.array(l[i])))
    return m


def Select(lt,p,r,n):
    if p == r:
        return lt[p]
    else:
        m = Divide(lt,p,r)
        x = Select(m,0,len(m)-1,int((len(m)-1)/2))
        q = Partition3(lt,p,r,x)
        if n == q:
            return lt[q]
        elif n < q:
            return Select(lt,p,q-1,n)
        else:
            return Select(lt,q+1,r,n)











# closest pair algorithm
def Mg(ls,p,q,r):
    p1 = copy.deepcopy(p)
    p2 = copy.deepcopy(q) + 1
    m = []
    while (p1<=q)&(p2<=r):
        if ls[p1][1] <= ls[p2][1]:
            m.append(ls[p1])
            p1 += 1
        else:
            m.append(ls[p2])
            p2 += 1
    if p1 == q+1:
        m.extend(ls[p2:r+1])
    else:
        m.extend(ls[p1:q+1])
    k = 0
    for i in range(p,r+1):
        ls[i] = m[k]
        k += 1
    return None


def Msort(ls,p,r):
    if p >= r:
        return None
    else:
        q = int((p+r)/2)
        Msort(ls,p,q)
        Msort(ls,q+1,r)
        Mg(ls,p,q,r)


def compute(x1,x2):
    return ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)**0.5


def find(x,l):
    n = len(x)
    point = [l[row[0]] for i,row in enumerate(x)]
    mindis = float('inf')
    minpair = []
    for i in range(n-1):
        for j in range(i+1,n):
            distance = compute(point[j],point[i])
            if distance < mindis:
                mindis = distance
                minpair = [l[x[i][0]],l[x[j][0]]]
    return minpair,mindis


def StripDis(x,l,d,y):
    point = []
    mindis = float('inf')
    minpair = []
    for i in range(len(y)):
        if l[y[i][0]][0] - x[math.ceil(len(x)/2)][1] <= d:
            point.append(y[i])

    for i in range(len(point)-1):
        x1 = l[point[i][0]]
        if i+8 < len(point):
            rear = i+8
        else:
            rear = len(point)
        for j in range(i+1,rear):
            x2 = l[point[j][0]]
            distance = compute(x1,x2)
            if distance < mindis:
                mindis = distance
                minpair = [x1,x2]
    return minpair,mindis


def Close(x,l,y):
    n = len(x)
    if n <= 4:
        return find(x,l)
    else:
        lpoint = x[0:math.ceil(n/2)]
        rpoint = x[math.ceil(n/2):]
        lp, ld = Close(lpoint,l,y)
        rp, rd = Close(rpoint,l,y)
        d = min(ld,rd)
        sp, sd = StripDis(x,l,d,y)
        mindis = min(d,sd)
        if mindis == ld:
            minpair = lp
        elif mindis == rd:
            minpair = rp
        else:
            minpair = sp
        return minpair,mindis


def ClosestPair(l):
    x = [[i,row[0]] for i,row in enumerate(l)]
    y = [[i,row[1]] for i,row in enumerate(l)]
    Msort(x,0,len(x)-1)
    Msort(y,0,len(y)-1)
    return Close(x,l,y)


def BruteClose(l):
    n = len(l)
    mindis = float('inf')
    minpair = []
    for i in range(n-1):
        for j in range(i+1,n):
            distance = compute(l[i], l[j])
            if distance < mindis:
                mindis = distance
                minpair = [l[i],l[j]]
    return minpair,mindis


if __name__ == '__main__':
    # ls = [6,5,4,7,9,21,1]
    # print(InsertSort(ls))
    # print(BubbleSort(ls))
    # print(SelectSort(ls))
    # lt = copy.deepcopy(ls)
    # QuickSort(lt,0,len(lt)-1)
    # lt1 = copy.deepcopy(ls)
    # MergeSort(lt1,0,len(lt1)-1)
    # print(lt)
    # print(lt1)
    # print(ls)
    # ls = np.arange(10,0,-1)
    # lt2 = copy.deepcopy(ls)
    # for i in range(len(ls)):
    #     print(Select(lt2,0,len(lt2)-1,i))
    # ls = 200 * (np.random.random((50,2))-0.5)
    # ls = np.random.randint(-50,50,size=[50,2])
    # ls = [[row[0],row[1]] for i,row in enumerate(ls)]
    # start1 = time.clock()
    # print(BruteClose(ls))
    # print(time.clock()-start1)
    # start2 = time.clock()
    # print(ClosestPair(ls))
    # print(time.clock()-start2)


