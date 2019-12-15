import numpy as np
import math
import random

# 1. compute fib number
# question a
def compute_exp(a,n):
    if n == 0:
        return 1
    elif n == 1:
        return a
    else:
        if n%2 == 0:
            b = compute_exp(a,n/2)
            return b*b
        else:
            b = compute_exp(a,(n-1)/2)
            return b * b * a


def compute_fib(n):
    a1 = (1 + 5 ** 0.5) / 2
    a2 = (1 - 5 ** 0.5) / 2
    part1 = compute_exp(a1,n)
    part2 = compute_exp(a2,n)
    return (part1 - part2)/5**0.5


# question b
def compute_matrix_exp(a,n):
    if n == 0:
        return np.array([[0,0],[0,0]])
    elif n == 1:
        return np.array([[1,1],[1,0]])
    else:
        if n%2 == 0:
            b = compute_matrix_exp(a,n/2)
            return np.dot(b,b)
        else:
            b = compute_matrix_exp(a,(n-1)/2)
            return np.dot(a,np.dot(b,b))


def compute_integer_fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    a = np.array([[1, 1], [1, 0]])
    b = compute_matrix_exp(a, n-1)
    return b[0,0]


# 2. matrix chain product


# 3. longest common subsequence


# 4. billboard placement
def place_billboard(x,r,m,t):
    c = np.zeros(m+1+t+1)
    p = np.zeros(m+1)
    direction = np.zeros(m+1)
    billboard = np.zeros(m+1)
    for i in range(m+1):
        direction[i] = i-1
        billboard[i] = float('nan')
    j = 0
    for i in range(m+1):
        if i == x[j]:
            p[i] = r[j]
            if j+1 < len(x):
                j += 1
            m1 = c[i-t-1] + p[i]
            m2 = c[i-1]
            if m1 > m2:
                c[i] = m1
                direction[i] = i-t-1
                billboard[i] = i
            else:
                c[i] = c[i-1]
                direction[i] = i-1
        else:
            c[i] = c[i-1]
            direction[i] = i-1

    # output result
    print('max revenue: %d' % c[m])
    i = m
    ls = []
    while i >= 0:
        if not math.isnan(billboard[i]):
            ls.append(billboard[i])
        i = int(direction[i])
    ls.reverse()
    print('location of billboards:')
    print(ls)


# 5. RNA secondary structure
def is_pair(a,b):
    if (a =='c')&(b =='g'):
        return True
    elif (a =='g')&(b =='c'):
        return True
    elif (a =='a')&(b =='t'):
        return True
    elif (a =='t')&(b =='a'):
        return True
    return False


def max_rna_pair(s):
    n = len(s)
    opt = np.zeros((n,n))
    v = np.zeros(n)
    node = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            node[i,j] = float('nan')
    for k in range(5,n):
        for i in range(n-k):
            j = i + k
            for t in range(i,j-4):
                if is_pair(s[t],s[j]):
                    v[t] = opt[i,t-1] + opt[t+1,j-1] + 1
                else:
                    v[t] = opt[i,j-1]
            vmax = max(v)
            pmax = np.argmax(v)
            v = np.zeros(n)
            if vmax > opt[i,j-1]:
                opt[i,j] = vmax
                node[i,j] = pmax
            else:
                opt[i,j] = opt[i,j-1]
                # if not math.isnan(node[i,j-1]):
                #     node[i,j] = node[i,j-1]
    print(opt[0,n-1])
    return node


def print_rna_pair(node,p,r,pair):
    if p == r:
        return None
    if not math.isnan(node[p,r]):
        t = int(node[p,r])
        pair.append([t,r])
        print_rna_pair(node,p,t-1,pair)
        print_rna_pair(node,t+1,r-1,pair)
    else:
        if r - 1 >= 0:
            print_rna_pair(node,p,r-1,pair)
        else:
            return None


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
                record[i,j,0] = i; record[i,j,1] = j
            else:
                c1 = a[j] - a[i]
                c2 = p[i+1,j]
                c3 = p[i,j-1]
                p[i,j] = max(c1,c2,c3)
                if p[i,j] == c1:
                    record[i,j,0] = i
                    record[i,j,1] = j
                elif p[i,j] == c2:
                    record[i,j,0] = record[i+1,j,0]
                    record[i,j,1] = record[i+1,j,1]
                elif p[i,j] == c3:
                    record[i,j,0] = record[i,j-1,0]
                    record[i,j,1] = record[i,j-1,1]
    return p,record


def k_shot_stock(a,num_k):
    p, record = one_shot_profit(a)
    n = len(a)
    m = np.zeros((n+1,num_k+1))
    bsmatrix = np.zeros((n+1,num_k+1))
    for k in range(1,num_k+1):
        for i in range(1,n+1):
            if k == 1:
                m[i,k] = p[1,i]
            else:
                v = np.zeros(n+1)
                for t in range(2*(k-1),i-2+1):
                    v[t] = max(m[t,k-1] + p[t+1,i],m[t,k])
                m[i,k] = max(v)
                day = np.argmax(v)
                bsmatrix[i, k] = day
    # get date
    d = []; t = n; i = num_k; pt = n
    while i > 0:
        t = int(bsmatrix[t,i])
        d.append([record[t+1,pt,0],record[t+1,pt,1]])
        pt = t
        i -= 1
    d.reverse()
    return m,d


# 7. coin change problem
def coin_change(c,a):
    n = len(c)
    c.reverse()
    c.insert(0,0)
    m = np.ones((n+1,a+1),dtype=int)
    for i in range(1,n+1):
        for j in range(1, a + 1):
            if (j == 1) | (i == 1):
                m[i,j] = 1
            else:
                if c[i] <= j:
                    m[i,j] = m[i-1,j] + m[i,j-c[i]]
                else:
                    m[i,j] = m[i-1,j]
    return m[n,a]


if __name__ == '__main__':
    np.set_printoptions(threshold=1e6)
    # test question 4
    # a = [i for i in range(101)]
    # x = random.sample(a,20)
    # x = np.sort(x)
    # r = np.random.randint(5,10,size=20)
    # m = 100
    # t = 5
    # place_billboard(x,r,m,t)

    # test question 5
    # s = 'acgtcgattcgagcgaatcgtacgaacgagcatagcggctagac'
    # node = max_rna_pair(s)
    # pair = []
    # print_rna_pair(node,0,len(s)-1,pair)
    # for i in range(len(pair)):
    #     print(pair[i])
        # print(s[pair[i][0]],s[pair[i][1]])

    # # test question 6
    # a = np.random.randint(1,50,100)
    # k = 10
    # m,d = k_shot_stock(a,k)
    # print(a)
    # print('max profit: %d' % m[len(a),k])
    # print(d)
    # for i in range(1,k+1):
    #     print('%d trade %d %d'%(i,a[d[i-1][0]-1],a[d[i-1][1]-1]))

    # test question 7
    a = 100
    c = [4,2,1]
    print(coin_change(c,a))
