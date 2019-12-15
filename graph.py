import numpy as np


class Node:
    def __init__(self,data=None,next=None):
        self.data = data
        self.next = next


def create_linklist(head_data=None):
    num = int(input('please input length of linklist: '))
    lk = Node()
    if head_data:
        lk.data = head_data
    for i in range(num):
        new_node = Node()
        new_node.data = int(input('content: '))
        new_node.next = lk.next
        lk.next = new_node
    return lk


def create_linklist2(head_data=None):
    length = int(input('please input degree of node: '))
    lk = Node()
    if head_data:
        lk.data = head_data
    rear = lk
    for i in range(length):
        new_node = Node()
        new_node.data = int(input('content: '))
        rear.next = new_node
        rear = new_node
    return lk


def print_linklist(lk):
    if lk.data:
        print(lk.data, end='')
    else:
        print('head', end='')
    node = lk.next
    while node:
        if node.data:
            print(' -> {}'.format(node.data), end='')
        node = node.next
    return None


def create_graph():
    num = int(input('please input number of node in graph: '))
    graph = [None for i in range(num+1)]
    for i in range(1,num+1):
        head_data = int(input('node: '))
        lk = create_linklist2(head_data)
        graph[head_data] = lk
    return graph


def print_graph(graph):
    for lk in graph:
        if lk:
            print_linklist(lk)
            print('')


def breadth_first_search(graph,start_node):
    # initialize parent, distance, and color list
    parent = [None for i in range(len(graph))]
    distance = [None for i in range(len(graph))]
    color = ['white' for i in range(len(graph))]
    color[0] = 'black'
    distance[start_node.data] = 0
    color[start_node.data] = 'grey'

    # create queue
    queue = []
    queue.append(start_node.data)
    while queue:
        prt = queue.pop(0)
        adjlist = graph[prt]
        node = adjlist.next
        while node:
            if color[node.data] == 'white':
                color[node.data] = 'grey'
                distance[node.data] = distance[prt] + 1
                parent[node.data] = prt
                queue.append(node.data)
            node = node.next
        color[prt] = 'black'

    bfs_tree = {}
    bfs_tree['distance'] = distance
    bfs_tree['color'] = color
    bfs_tree['parent'] = parent
    bfs_tree['start_node'] = start_node.data
    return bfs_tree


def print_bfs_spanning_tree(bfs_tree):
    parent = bfs_tree['parent']
    n = len(parent)
    for i in range(1,n):
        if parent[i]:
            print('{} -> '.format(i), end='')
        else:
            print('{} '.format(i), end='')
        nd = parent[i]
        while nd:
            if parent[nd]:
                print('{} -> '.format(nd), end='')
            else:
                print('{} '.format(nd), end='')
            nd = parent[nd]
        print('')
    return None


if __name__ == '__main__':
    print('')
    print('-'*30)
    graph = create_graph()
    print_graph(graph)
    print('')
    print('-' * 30)
    bfs_tree = breadth_first_search(graph, graph[1])
    print_bfs_spanning_tree(bfs_tree)
