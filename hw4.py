import numpy as np
import random

# 3. cellphone station
def minimum_station(x):
    station = [None]
    start = x[0]
    flag = 0
    i = 0
    j = 0
    n = len(x)
    while i < n:
        if flag == 0:
            if abs(x[i] - start) <= 5:
                station[j] = x[i]
                i += 1
            else:
                flag = 1
        else:
            if abs(x[i] - station[j]) <= 5:
                i += 1
            else:
                start = x[i]
                station.append(None)
                j += 1
                flag = 0
    return station


def interval_color(v):
    v = sorted(v,key= lambda x:x[0])
    queue = []
    f_time = []
    for i in range(len(v)):
        flag = 0
        for j in range(len(queue)):
            if v[i][0] >= f_time[j]:
                f_time[j] = v[i][1]
                queue[j].append(i)
                flag = 1
        if not flag:
            queue.append([i])
            f_time.append(v[i][1])
    print(queue)
    print(len(queue))
    return len(queue)


if __name__ == '__main__':
    # test 3
    # x = [i for i in range(50)]
    # x = random.sample(x,10)
    # x.sort()
    # print(x)
    # station = minimum_station(x)
    # print(station)

    # test interval_color
    v = [[0, 30],[5, 10],[15, 20]]
    interval_color(v)


