import numpy as np

def create_graph(weight=None):
    node_num = int(input('please input number of node: '))
    graph = [node_num]
    for i in range(1,node_num+1):
        ls = []
        node = int(input('node: '))
        ls.append(node)
        count = int(input('degree: '))
        for j in range(count):
            ele = int(input('neighbor: '))
            if weight:
                w = float(input('weight: '))
                ele = [ele,w]
            ls.append(ele)
        graph.append(ls)
    return graph


def print_graph(graph,weight=None):
    n = len(graph)
    for i in range(1,n):
        print(graph[i][0],end='')
        adjls = graph[i][1:]
        for j in adjls:
            if not weight:
                print(' -> {}'.format(j),end='')
            else:
                print(' -> {}'.format(j[0]),end='')
        print('')
    return None


def breadth_first_search(graph,start_node):
    # initialize arrays of nodes
    color = ['white' for i in graph]
    dis = [None for i in graph]
    parent = [None for i in graph]
    color[0] = 'black'
    color[start_node] = 'grey'
    dis[start_node] = 0

    # create a queue, start traverse
    queue = []
    queue.append(start_node)
    while queue:
        prt = queue.pop(0)
        adjls = graph[prt][1:]
        for son in adjls:
            if color[son] == 'white':
                color[son] = 'grey'
                dis[son] = dis[prt] + 1
                parent[son] = prt
                queue.append(son)
        color[prt] = 'black'

    # return result
    bfs_tree = {}
    bfs_tree['dis'] = dis
    bfs_tree['parent'] = parent
    bfs_tree['color'] = color
    bfs_tree['start_node'] = start_node
    return bfs_tree


def print_search_tree(tree):
    parent = tree['parent']
    for node in range(1,len(parent)):
        print(node, end='')
        while parent[node]:
            print(' <- {}'.format(parent[node]),end='')
            node = parent[node]
        print('')
    return None


def simple_bfs(graph,start_node,color):
    queue = []
    queue.append(start_node)
    while queue:
        prt = queue.pop(0)
        adjls = graph[prt][1:]
        for son in adjls:
            if color[son] == 'white':
                queue.append(son)
                color[son] = 'grey'
        color[prt] = 'black'


def connectivity_test(graph):
    # initialzie array
    color = ['white' for i in graph]
    color[0] = 'black'
    count = 0
    for i in range(1,len(graph)):
        if color[i] == 'white':
            simple_bfs(graph,i,color)
            count += 1
            print(color)
    if count == 1:
        print('connected')
    else:
        print('unconnected')
    return None


class Vertex:
    def __init__(self,name,distance):
        self.distance = distance
        self.name = name


class VertexMinHeap:
    def __init__(self):
        self.item_list = [0]
        self.position = {}

    def insert(self,new_vertex):
        self.item_list.append(new_vertex)
        self.item_list[0] += 1
        self.position[new_vertex.name] = self.item_list[0]
        self.upward_adjust(self.item_list[0])

    def swap_position(self,i1,i2):
        self.position[self.item_list[i1].name] = i2
        self.position[self.item_list[i2].name] = i1


    def upward_adjust(self,index):
        while index//2 > 0:
            if self.item_list[index].distance < self.item_list[index//2].distance:
                self.item_list[index], self.item_list[index//2] = self.item_list[index//2],self.item_list[index]
                self.swap_position(index,index//2)
                index //= 2
            else:
                break

    def get_min(self):
        if self.item_list[0] == 0:
            return None
        else:
            return self.item_list[1]

    def pop_min(self):
        if self.item_list[0] == 0:
            return None
        else:
            value = self.item_list[1]
            self.position[self.item_list[self.item_list[0]].name] = 1
            self.item_list[1] = self.item_list[self.item_list[0]]
            self.item_list[0] -= 1
            self.downward_adjust(1)
            return value


    def downward_adjust(self,index):
        while 2*index <= self.item_list[0]:
            if 2*index + 1 <= self.item_list[0]:
            # both children exist
                if self.item_list[2*index].distance < self.item_list[2*index+1].distance:
                # l < r
                    if self.item_list[index].distance > self.item_list[2*index].distance:
                        self.swap_position(index, 2 * index)
                        self.item_list[index],self.item_list[2 * index] = self.item_list[2*index],self.item_list[index]
                        index = index * 2
                    else:
                        break
                else:
                # r <= l
                    if self.item_list[index].distance > self.item_list[2*index+1].distance:
                        self.swap_position(index, 2 * index + 1)
                        self.item_list[index],self.item_list[2 * index + 1] = self.item_list[2*index+1],self.item_list[index]
                        index = index * 2 + 1
                    else:
                        break
            else:
            # left child exist
                if self.item_list[index].distance > self.item_list[2*index].distance:
                    self.swap_position(index, 2 * index + 1)
                    self.item_list[index], self.item_list[2 * index] = self.item_list[2 * index], self.item_list[index]
                    index = 2 * index
                else:
                    break

    def is_empty(self):
        return self.item_list[0] == 0

    def size(self):
        return self.item_list[0]

    def build_heap(self,input_list):
        self.item_list.extend(input_list)
        self.item_list[0] = len(input_list)
        for i in range(self.item_list[0]):
            self.position[input_list[i].name] = i + 1
        # print(self.position)
        for i in range(self.item_list[0]//2,0,-1):
            self.downward_adjust(i)
        # print(self.position)


def relax(u,v,prt,du,dv,x):
    if du + x < dv:
        dv = du + x
        prt[v] = u
        print('??')
    return dv


def djk_shortest_path(graph,start_node,weight):
    # initialize
    vertices = [Vertex(i,5-i) for i in range(1,graph[0]+1)]
    vertices[start_node-1].distance = 0
    prt = {}
    q = VertexMinHeap()
    q.build_heap(vertices)
    s = {}
    while q.size() > 0:
        du = q.get_min().distance
        u = q.get_min().name
        q.pop_min()
        s[u] = 1
        print(s)
        adj = graph[u][1:]
        for v in adj:
            print(q.position)
            if v not in s:
                print('p: {}'.format(q.position[v]))
                dv = q.item_list[q.position[v]].distance
                q.item_list[q.position[v]].distance = relax(u,v,prt,dv,du,weight[u,v])
                print(dv)
                print(q.item_list[q.position[v]].distance)
                print('-'*20)
    return prt



# DFS DFS DFS

def dfs_visit(graph,node,color,parent,df,time,edge_color,sort_list,component,d_sort):
    color[node] = 'grey'
    time += 1
    df[node][0] = time
    d_sort.append(node)
    adjls = graph[node][1:]
    for son in adjls:
        key = str(node) + '-' + str(son)
        if key not in edge_color.keys():
            edge_color[key] = color[son]
        if color[son] == 'white':
            parent[son] = node
            time = dfs_visit(graph,son,color,parent,df,time,edge_color,sort_list,component,d_sort)

    color[node] = 'black'
    time += 1
    df[node][1] = time
    sort_list.insert(0,node)
    component.insert(0,node)
    return time


def deepth_first_search(graph,start_node):
    # initialize arrays
    color = ['white' for i in graph]
    color[0] = 'black'
    edge_color = {}
    parent = [None for i in graph]
    df = [[0,0] for i in graph]
    sort_list = []
    d_sort = []
    time = 0
    component = []
    dfs_visit(graph,start_node,color,parent,df,time,edge_color,sort_list,component,d_sort)

    dfs_tree = {}
    dfs_tree['start_node'] = start_node
    dfs_tree['parent'] = parent
    dfs_tree['color'] = color
    dfs_tree['df'] = df
    dfs_tree['edge_color'] = edge_color
    dfs_tree['sort_list'] = sort_list
    for i in edge_color:
        if edge_color[i] == 'grey':
            print('graph has directed circle')
    return dfs_tree


def get_dfs_forest(graph,order=None):
    color = ['white' for i in graph]
    color[0] = 'black'
    edge_color = {}
    parent = [None for i in graph]
    df = [[0,0] for i in graph]
    time = 0
    sort_list = []
    d_sort = []
    dfs_forest = {}
    dfs_forest['component'] = []
    if not order:
        for i in range(1,len(graph)):
            component = []
            if color[i] == 'white':
                time = dfs_visit(graph,i,color,parent,df,time,edge_color,sort_list,component,d_sort)
                dfs_forest['component'].append(component)
    else:
        for i in order:
            component = []
            if color[i] == 'white':
                time = dfs_visit(graph,i,color,parent,df,time,edge_color,sort_list,component,d_sort)
                dfs_forest['component'].append(component)

    dfs_forest['parent'] = parent
    dfs_forest['color'] = color
    dfs_forest['df'] = df
    dfs_forest['edge_color'] = edge_color
    dfs_forest['sort_list'] = sort_list
    dfs_forest['d_sort'] = d_sort
    return dfs_forest


def transpose_graph(graph):
    trans_graph = [[i] for i in range(len(graph))]
    for i in range(1,len(graph)):
        adjls = graph[i][1:]
        for j in adjls:
            trans_graph[j].append(i)
    return trans_graph


def strong_connectivity_test(graph):
    dfs_forest = get_dfs_forest(graph)
    sort_verticies = dfs_forest['sort_list']
    trans_graph = transpose_graph(graph)
    dfs_forest2 = get_dfs_forest(trans_graph,sort_verticies)
    return dfs_forest2['component']


def get_son(dfs_forest):
    tree_set = {}
    parent = dfs_forest['parent']
    for component in dfs_forest['component']:
        for v in component:
            node = v
            tree = [node]
            while parent[node]:
                tree.insert(0,parent[node])
                node = parent[node]
            tree_set[str(v)] = tree
    return tree_set


def biconnected_test(graph):
    dfs_forest = get_dfs_forest(graph)
    d_sort = dfs_forest['d_sort']
    df = dfs_forest['df']
    edge_color = dfs_forest['edge_color']
    parent = dfs_forest['parent']
    low = [0 for i in graph]



def bellman_ford(graph):
    pass


if __name__ == '__main__':
    graph = create_graph()
    weight = np.array([[0,0,0,0,0,0],
                       [0,0,5,10,float('inf'),float('inf')],
                       [0,float('inf'),0,3,2,9],
                       [0,float('inf'),2,0,float('inf'),1],
                       [0,float('inf'),float('inf'),float('inf'),0,6],
                       [0,float('inf'),float('inf'),float('inf'),4,0]])
    # print_graph(graph)
    # print('-'*30)
    # prt = djk_shortest_path(graph,1,weight)
    # for i in range(1,len(prt)):
    #     print(i,':',prt[i])
    # bfs_tree = breadth_first_search(graph,1)
    # print_search_tree(bfs_tree)
    print('-' * 30)
    connectivity_test(graph)
    #
    # dfs_tree = deepth_first_search(graph,1)
    # print_search_tree(dfs_tree)
    # print('-' * 30)
    #
    # component = strong_connectivity_test(graph)
    # print(component)
