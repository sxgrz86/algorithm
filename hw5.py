import numpy as np
from heapq import *
from copy import deepcopy

def create_graph():
    number = int(input('please input number of node: '))
    graph = [number]
    for i in range(1,number+1):
        ls = [int(input('node: '))]
        deg = int(input('degree: '))
        for j in range(1,deg+1):
            neighbor = int(input('neighbor: '))
            ls.append(neighbor)
        graph.append(ls)
    return graph


def print_graph(graph):
    for i in range(1,len(graph)):
        print(i,end='')
        for j in range(1,len(graph[i])):
            print(' -> {}'.format(graph[i][j]),end='')
        print()


class Vertex:
    def __init__(self,name):
        self.name = name
        self.value = float('inf')


class VertexMinHeap:
    def __init__(self):
        self.item_list = [0]
        self.position = {}

    def upward_adjust(self,index):
        while index//2:
            if self.item_list[index].value < self.item_list[index//2].value:
                p = self.item_list[index // 2]
                s = self.item_list[index]
                self.position[p.name] = index
                self.position[s.name] = index // 2
                self.item_list[index], self.item_list[index//2] = self.item_list[index//2], self.item_list[index]
                index = index//2
            else:
                break

    def downward_adjust(self,index):
        while 2*index <= self.item_list[0]:
            # have leaf
            if 2*index + 1 <= self.item_list[0]:
            # 2 leaves
                if self.item_list[2*index].value < self.item_list[2*index+1].value:
                # l < r
                    if self.item_list[2*index].value < self.item_list[index].value:
                        p = self.item_list[index]
                        s = self.item_list[2 * index]
                        self.position[p.name] = 2 * index
                        self.position[s.name] = index
                        self.item_list[2 * index], self.item_list[index] = self.item_list[index], self.item_list[2*index]
                        index *= 2
                    else:
                        break

                else:
                # r <= l
                    if self.item_list[2*index+1].value < self.item_list[index].value:
                        p = self.item_list[index]
                        s = self.item_list[2 * index + 1]
                        self.position[p.name] = 2 * index + 1
                        self.position[s.name] = index
                        self.item_list[2 * index + 1],self.item_list[index] = self.item_list[index],self.item_list[2*index+1]
                        index = index * 2 + 1
                    else:
                        break
            else:
            # 1 leaf
                if self.item_list[2 * index].value < self.item_list[index].value:
                    p = self.item_list[index]
                    s = self.item_list[2 * index]
                    self.position[p.name] = 2 * index
                    self.position[s.name] = index
                    self.item_list[2 * index], self.item_list[index] = self.item_list[index], self.item_list[2 * index]
                    index *= 2
                else:
                    break

    def pop(self):
        if self.item_list[0] > 0:
            vt = self.item_list[1]
            n = self.item_list[0]
            self.item_list[1],self.item_list[n] = self.item_list[n],self.item_list[1]
            self.item_list[0] -= 1

            v1 = self.item_list[1]
            v2 = self.item_list[n]
            self.position[v1.name] = 1
            self.position[v2.name] = n
            self.downward_adjust(1)
            return vt
        else:
            return None

    def append(self,vertex):
        self.item_list[0] += 1
        self.item_list.append(vertex)
        n = self.item_list[0]
        self.upward_adjust(n)

    def size(self):
        return self.item_list[0]

    def build_heap(self,name_list,start_node):
        # make a vertex list
        ls = []
        for i in range(len(name_list)):
            v = Vertex(name_list[i])
            if v.name == start_node:
                v.value = 0
            self.position[v.name] = i+1
            ls.append(v)
        self.item_list.extend(ls)
        self.item_list[0] += len(ls)
        last_index = self.item_list[0]//2
        for i in range(last_index,0,-1):
            self.downward_adjust(i)


def relax(u,v,weight,hp,parent):
    pu = hp.position[u]
    pv = hp.position[v]
    new_dis = hp.item_list[pu].value + weight[u][v]
    if new_dis < hp.item_list[pv].value:
        hp.item_list[pv].value = new_dis
        hp.upward_adjust(pv)
        parent[v] = u


def djk(graph,weight):
    hp = VertexMinHeap()
    # start node is 1
    hp.build_heap([1,2,3,4,5],1)
    s = []
    parent = {1:None}
    while hp.size():
        u = hp.pop().name
        s.append(u)
        # print(s)
        adj = graph[u][1:]
        for v in adj:
            if v not in s:
                relax(u,v,weight,hp,parent)
        # print(hp.position)
    return parent


def relax2(u,v,weight,distance,parent):
    new_dis = distance[u] + weight[u][v]
    if new_dis < distance[v]:
        distance[v] = new_dis
        parent[v] = u


def bellman_ford_algorithm(graph,weight):
    # initialize
    distance = {}
    for i in range(1,graph[0]+1):
        distance[i] = float('inf')
    distance[1] = 0
    parent = {1:None}

    for i in range(1,graph[0]+1):
        for u in range(1,graph[0]+1):
            adj = graph[u][1:]
            for v in adj:
                relax2(u,v,weight,distance,parent)

    for u in range(1, graph[0] + 1):
        adj = graph[u][1:]
        for v in adj:
            if distance[v] > distance[u] + weight[u][v]:
                print('negative cycle exists')
                return None
    return parent





def operate(m1,m2):
    m3 = deepcopy(m1)
    for i in range(1,len(m1)):
        for j in range(1,len(m1)):
            min_dis = float('inf')
            for k in range(1,len(m1)):
                min_dis = min(m1[i][k] + m2[k][j],min_dis)
            m3[i][j] = min_dis
    return m3


def compute_mat(w,n):
    if n == 1:
        return w
    else:
        if n%2 == 0:
            a = compute_mat(w,n/2)
            return operate(a,a)
        else:
            a = compute_mat(w,n//2)
            b = operate(a,a)
            return operate(b,w)


def all_path_algorithm(weight,n):
    w = deepcopy(weight)
    return compute_mat(w,n)


def all2(weight,n):
    d1 = deepcopy(weight)
    d2 = deepcopy(weight)
    for i in range(2,n+1):
        d2 = operate(d1,weight)
        d1 = d2
    return d2


def operate2(m1,t):
    m2 = deepcopy(m1)
    for i in range(1,len(m1)):
        for j in range(1,len(m1)):
            m2[i][j] = min(m1[i][t] + m1[t][j], m1[i][j])
    return m2


def floyd_algorithm(weight,n):
    d0 = deepcopy(weight)
    dn = deepcopy(weight)
    for i in range(1,n+1):
        dn = operate2(d0,i)
        d0 = dn
    return dn


def set_weight():
    weight = [[-1]*6,[-1,0,10,5,-1,-1],[-1,-1,0,2,1,-1],[-1,-1,3,0,9,2],[-1,-1,-1,-1,0,4],[-1,7,-1,-1,6,0]]
    for i in range(len(weight)):
        for j in range(len(weight[0])):
            if weight[i][j] == -1:
                weight[i][j] = float('inf')
    return weight


if __name__ == '__main__':
    print('')
    #test shortest path algorithm
    graph = [5,[1,2,3],[2,3,4],[3,2,4,5],[4,5],[5,1,4]]
    weight = set_weight()
    print(weight)
    print('-'*20)
    print('djk {}'.format(djk(graph,weight)))
    print('bellman: {}'.format(bellman_ford_algorithm(graph,weight)))
    print('-'*20)
    print('d&c all path: {}'.format(all_path_algorithm(weight,graph[0])[1:]))
    print('-'*20)
    print('simple all path: {}'.format(all2(weight,graph[0])[1:]))
    print('-' * 20)
    print('floyd: {}'.format(floyd_algorithm(weight,graph[0])[1:]))
    print('-' * 20)





