import numpy as np
import math
import random


# 1. compute fib
# question a
def compute_exp(a,n):
    if n == 0:
        return 1
    elif n == 1:
        return a
    else:
        if n%2 == 0:
            t = compute_exp(a,n/2)
            return t*t
        else:
            t = compute_exp(a,(n-1)/2)
            return t*t*a


def compute_fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        part1 = compute_exp((1+5**0.5)/2,n)
        part2 = compute_exp((1-5**0.5)/2,n)
        return (part1 - part2)/5**0.5


# question b
def compute_matrix_exp(a,n):
    if n == 1:
        return a
    elif n%2 == 0:
        t = compute_matrix_exp(a,n/2)
        return np.dot(t,t)
    else:
        t = compute_matrix_exp(a,(n-1)/2)
        return np.dot(np.dot(t,t),a)


def compute_integer_fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a = [[1,1],[1,0]]
        matrix = compute_matrix_exp(a,n-1)
        return matrix[0][0]


# 2. matrix chain product

# 3. longest common subsequence

# 4. billboard placement
def place_billboard(x,r,m,dis):
    n = len(x)
    c = np.zeros(m+1+dis+1)
    direction = np.zeros(m+1)
    k = 0
    for i in range(m+1):
        if i == x[k]:
            num = k
            if k+1 <= n-1:
                k += 1
            c[i] = max(c[i - 1], c[i - dis - 1] + r[num])
            if c[i] == c[i-1]:
                direction[i] = i - 1
            else:
                direction[i] = i - dis - 1
        else:
            c[i] = c[i - 1]
            direction[i] = i - 1

    # print billboard position
    i = m
    position = []
    while i >= 0:
        if direction[i] != i-1:
            position.append(i)
        i = int(direction[i])
    position.reverse()
    print('max revenue: %d' % c[m])
    print('loaction of billboards:')
    print(position)
    return c


def place_billboard2(x,r,dis):
    n = len(x)
    c = np.zeros(n+1)
    direction = np.zeros(n+1,dtype=int)
    for i in range(n):
        if i == 0:
            c[i] = r[i]
            direction[i] = -1
        else:
            p = r[i]
            last_position = -1
            for j in range(1,dis+2):
                if i - j >= 0:
                    if x[i] - x[i-j] > dis:
                        p = c[i-j] + r[i]
                        last_position = i - j
                        break
            c[i] = max(c[i-1],p)
            if c[i] == p:
                direction[i] = last_position
                if i == n-1:
                    direction[n] = i
            else:
                direction[i] = direction[i-1]
                if i == n-1:
                    direction[n] = direction[i-1]
    billboard = []
    i = n
    while i >= 0:
        i = direction[i]
        if (i >= 0) & (i != i-1):
            billboard.append(x[i])
    billboard.reverse()
    return c[n-1],billboard


# 5. RNA secondary structure
def is_pair(a,b):
    if (a == 'c') & (b == 'g'):
        return True
    elif (a == 'g') & (b == 'c'):
        return True
    elif (a == 'a') & (b == 't'):
        return True
    elif (a == 't') & (b == 'a'):
        return True


def get_rna_pairs(direction,p,r,pair):
    if r - p < 5:
        return None
    else:
        if not math.isnan(direction[p,r]):
            q = int(direction[p,r])
            if q == r-1:
                get_rna_pairs(direction, p, q, pair)
            else:
                pair.append([q, r])
                get_rna_pairs(direction, p, q - 1, pair)
                get_rna_pairs(direction, q + 1, r - 1, pair)


def max_rna_pairs(s):
    n = len(s)
    opt = np.zeros((n,n),dtype=int)
    direction = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            direction[i,j] = float('nan')
    for k in range(5,n):
        for i in range(n-k):
            j = i + k
            v = np.zeros(n,dtype=int)
            for t in range(i,j-4):
                if is_pair(s[t],s[j]):
                    v[t] = opt[i,t-1] + opt[t+1,j-1] + 1
            vmax = max(v)
            index = np.argmax(v)
            if vmax > opt[i,j-1]:
                opt[i,j] = vmax
                direction[i,j] = index
            else:
                opt[i,j] = opt[i,j-1]
                direction[i,j] = j-1

    # print pairs
    pair = []
    get_rna_pairs(direction,0,n-1,pair)
    print('max pair number:%d' % opt[0,n-1])
    for i in range(len(pair)):
        print(pair[i])
        print(s[pair[i][0]],s[pair[i][1]])
    return opt,pair


# 6. k-shot stock
def one_shot_profit(a):
    n = len(a)
    a = np.insert(a,0,0)
    p = np.zeros((n+1,n+1))
    record = np.zeros((n+1,n+1,2),dtype=int)
    for k in range(1,n):
        for i in range(1,n-k+1):
            j = i + k
            if k == 1:
                p[i,j] = a[j] - a[i]
                record[i,j,0] = i
                record[i,j,1] = j
            else:
                c1 = a[j] - a[i]
                c2 = p[i,j-1]
                c3 = p[i+1,j]
                p[i,j] = max(c1,c2,c3)
                if p[i,j] == c1:
                    record[i, j, 0] = i
                    record[i, j, 1] = j
                elif p[i,j] == c2:
                    record[i, j, 0] = record[i,j-1,0]
                    record[i, j, 1] = record[i,j-1,1]
                else:
                    record[i, j, 0] = record[i+1, j, 0]
                    record[i, j, 1] = record[i+1, j, 1]
    a = np.delete(a,0)
    return p,record


def k_shot_profit(a,k_num):
    n = len(a)
    p,record = one_shot_profit(a)
    m = np.zeros((n+1,k_num+1))
    bsmatrix = np.zeros((n+1,k_num+1))
    for k in range(k_num+1):
        for i in range(n+1):
            if k == 1:
                m[i,k] = p[1,i]
            else:
                v = np.zeros(n)
                for t in range(2*(k-1),i-1):
                    v[t] = m[t,k-1] + p[t+1,i]
                m[i,k] = max(v)
                index = np.argmax(v)
                bsmatrix[i,k] = index

    # get date
    p1 = n; pair = []; i = k_num
    while i > 0:
        d1 = int(bsmatrix[p1,i])
        pair.append([record[d1+1,p1,0],record[d1+1,p1,1]])
        p1 = d1
        i -= 1
    pair.reverse()
    return m,pair


# 7. coin change problem
def coin_change(c,a):
    n = len(c)
    c.reverse()
    c = np.insert(c,0,0)
    m = np.zeros((n+1,a+1),dtype=int)
    for i in range(1,n+1):
        for j in range(a+1):
            if j == 0:
                m[i,j] = 1
            else:
                if c[i] > j:
                    m[i,j] = m[i-1,j]
                else:
                    m[i,j] = m[i-1,j] + m[i,j-c[i]]
    return m[n,a]


if __name__ == '__main__':
    np.set_printoptions(threshold=1e6)
    # # test question 1
    # for i in range(10):
    #     print(compute_fib(i))
    # for i in range(10):
    #     print(compute_integer_fib(i))

    # test question 4
    a = [i for i in range(101)]
    x = random.sample(a,20)
    x = np.sort(x)
    r = np.random.randint(5,10,size=20)
    m = 100
    t = 5
    place_billboard(x,r,m,t)

    mp,bb = place_billboard2(x,r,t)
    print('-'*30)
    print('max revenue: {}'.format(mp))
    print('billboard: {}'.format(bb))
    j = 0
    m = 0
    for i in range(len(x)):
        if x[i] == bb[j]:
            print(x[i])
            m += r[i]
            j += 1
    print(m)
    # test question 5
    # s = 'acgtcgattcgagcgaatcgtacgaacgagcatagcggctagac'
    # max_rna_pairs(s)

    # test question 6
    # a = np.random.randint(1,50,100)
    # k = 10
    # m,d = k_shot_profit(a,k)
    # p = 0
    # for i in range(k):
    #     print('%d trade %d %d'%(i,a[d[i][0]-1],a[d[i][1]-1]))
    #     p += a[d[i][1]-1] - a[d[i][0]-1]
    # print('max profit: %d' % p)

    # test question 7
    # a = 100
    # c = [4,2,1]
    # print(coin_change(c,a))

