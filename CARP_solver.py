import argparse
import multiprocessing
import random
import time
from multiprocessing import Process, Pool, Lock, Queue
import concurrent.futures
import copy
import numpy as np

INT_INF = np.iinfo(np.int32).max
st = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('instance')
parser.add_argument('-t', type=int)
parser.add_argument('-s', type=int)

args = parser.parse_args()

with open(args.instance, 'r') as f:
    lines = f.readlines()
    lines = [i.rstrip() for i in lines]

header_vals = [int(i.split(" : ")[1]) for i in lines[1:8]]
v_cnt = header_vals[0]
depot = header_vals[1]
r_e_cnt = header_vals[2]
n_e_cnt = header_vals[3]
capacity = header_vals[5]
tc = header_vals[6]
td = 0

demands = []

cost_mat = np.full(shape=(v_cnt + 1, v_cnt + 1), dtype=int, fill_value=INT_INF)
np.fill_diagonal(cost_mat, 0)
demand_mat = np.zeros((v_cnt + 1, v_cnt + 1), dtype=int)

for i in lines[9:-1]:
    nums = i.split()
    v0 = int(nums[0])
    v1 = int(nums[1])
    cost_mat[v0][v1] = cost_mat[v1][v0] = int(nums[2])
    d = int(nums[3])
    if d > 0:
        td += d
        demands.append((v0, v1, d))
        demands.append((v1, v0, d))
        demand_mat[v0, v1] = demand_mat[v1, v0] = d


def minimal_distance(cost_mat):
    dis = cost_mat.copy()
    n = dis.shape[0]
    for k in range(1, n):
        for i in range(1, n):
            for j in range(1, n):
                if dis[i][j] > dis[i][k] + dis[k][j]:
                    dis[i][j] = dis[i][k] + dis[k][j]
    return dis


def near(v, demands_invalid, dis_mat):
    factor = tc / r_e_cnt
    res = []
    nea_tdx2 = 0

    for idx, demand in enumerate(demands):
        if demands_invalid[idx]:
            continue
        v0, v1, dem = demand
        if dis_mat[v, v0] <= factor or dis_mat[v, v1] <= factor:
            res.append(demand + (idx,))
            nea_tdx2 += dem

    return res, nea_tdx2


def path_scanning(cost_mat, dis_mat, demands):
    # hyper parameters
    alpha = 2.0

    k = 0
    rs = []
    total_cost = 0
    demands_invalid = np.full(len(demands), fill_value=False, dtype=bool)
    demands_valid_cnt = len(demands)

    while demands_valid_cnt:
        rsk = []
        rs.append(rsk)
        cost = 0
        rvc = capacity
        path_end = depot
        eff = 0

        while demands_valid_cnt:
            d = INT_INF
            candidates = []
            nea, nea_tdx2 = near(path_end, demands_invalid, dis_mat)
            use_rule = False

            if len(nea) and rvc <= alpha * nea_tdx2 / len(nea):
                use_rule = True
            elif len(nea) == 0 and rvc <= alpha * td / r_e_cnt:
                use_rule = True

            if use_rule:
                for idx, demand in enumerate(demands):
                    if demands_invalid[idx]:
                        continue

                    v0, v1, dem = demand
                    if dem > rvc:
                        continue

                    # efficiency rule
                    ell = (dis_mat[path_end, v0] + cost_mat[v0, v1] + dis_mat[v1, depot] - dis_mat[path_end, depot])

                    if ell > 0 and dem / ell < eff:
                        continue

                    if dis_mat[path_end][v0] < d:
                        candidates = [idx]
                        d = dis_mat[path_end][v0]
                    elif dis_mat[path_end][v0] == d:
                        candidates.append(idx)
            else:
                for idx, demand in enumerate(demands):
                    if demands_invalid[idx]:
                        continue

                    v0, v1, dem = demand
                    if dem > rvc:
                        continue

                    if dis_mat[path_end][v0] < d:
                        candidates = [idx]
                        d = dis_mat[path_end][v0]
                    elif dis_mat[path_end][v0] == d:
                        candidates.append(idx)

            if d == INT_INF:
                break

            cdem_idx = random.choice(candidates)
            cdem = demands[cdem_idx]

            rsk.append(cdem)
            path_end = cdem[1]
            rvc -= cdem[2]
            demands_invalid[cdem_idx | 1] = True
            demands_invalid[cdem_idx >> 1 << 1] = True
            cost += d + cost_mat[cdem[0]][cdem[1]]

            eff = (capacity - rvc) / (cost + dis_mat[path_end, depot])

            demands_valid_cnt -= 2

        cost += dis_mat[path_end][depot]
        total_cost += cost
        k += 1

    return rs, total_cost


def output(rs, total_cost):
    strs = []
    for r in rs:
        strs.append("0," + ",".join(f"({i[0]},{i[1]})" for i in r) + ",0")
    print("s", ",".join(strs))
    print(f"q {total_cost}")





# 0,0, xxxx, 0,0
# def filp(rs, rs_cost, rs_violation, dis_mat):
#     rs_copy_violation = rs_violation

#     for j in range(0, len(rs)):
#         for i in range(1, len(rs[j]) - 1):
#             rs_copy = copy.copy(rs)
#             rs_copy[j] = r = copy.copy(rs[j])
#             rs_copy_cost = rs_cost - dis_mat[r[i - 1][1]][r[i][0]] - dis_mat[r[i][1]][r[i + 1][0]] \
#                            + dis_mat[r[i - 1][1]][r[i][1]] + dis_mat[r[i][0]][r[i + 1][0]]
#             r[i] = (r[i][1], r[i][0])
#             yield rs_copy, rs_copy_cost, rs_copy_violation


def single_insertion(rs, rs_cost, rs_violation, dis_mat):
    # single insertion in the same route

    rs_copy_violation = rs_violation
    for k in range(0, len(rs)):
        for i in range(1, len(rs[k]) - 1):
            for j in range(1, len(rs[k])):
                if i + 1 == j or i == j:
                    continue
                rs_copy = copy.copy(rs)
                rs_copy[k] = r = copy.copy(rs[k])
                rs_copy_cost = rs_cost - dis_mat[r[i - 1][1]][r[i][0]] - dis_mat[r[i][1]][r[i + 1][0]] \
                                       + dis_mat[r[i - 1][1]][r[i + 1][0]] \
                                       - dis_mat[r[j - 1][1]][r[j][0]] \
                                       + dis_mat[r[j - 1][1]][r[i][0]] + dis_mat[r[i][1]][r[j][0]]
                ri = r.pop(i)
                if i < j:
                    r.insert(j - 1, ri)
                else:
                    r.insert(j, ri)
                # print(i, j)
                yield rs_copy, rs_copy_cost, rs_copy_violation


                rs_copy = copy.copy(rs_copy)
                r = rs[k]
                rs_copy[k] = copy.copy(rs_copy[k])
                rs_copy_cost = rs_copy_cost - dis_mat[r[j - 1][1]][r[i][0]] - dis_mat[r[i][1]][r[j][0]] \
                                            + dis_mat[r[j - 1][1]][r[i][1]] + dis_mat[r[i][0]][r[j][0]]
                if i < j:
                    rs_copy[k][j - 1] = (r[i][1], r[i][0])
                else:
                    rs_copy[k][j] = (r[i][1], r[i][0])
                yield rs_copy, rs_copy_cost, rs_copy_violation

    # single insertion across routes
    for i in range(0, len(rs)):
        d0 = sum([demand_mat[n[0]][n[1]] for n in rs[i]])
        for k in range(0, len(rs)):
            if (i == k):
                continue
            d1 = sum([demand_mat[n[0]][n[1]] for n in rs[k]])
            for j in range(1, len(rs[i]) - 1):
                for t in range(1, len(rs[k])):
                    rs_copy = copy.copy(rs)
                    r0 = rs_copy[i] = copy.copy(rs[i])
                    r1 = rs_copy[k] = copy.copy(rs[k])

                    rs_copy_cost = rs_cost - dis_mat[r0[j - 1][1]][r0[j][0]] - dis_mat[r0[j][1]][r0[j + 1][0]] \
                                   - dis_mat[r1[t - 1][1]][r1[t][0]] \
                                   + dis_mat[r0[j - 1][1]][r0[j + 1][0]] \
                                   + dis_mat[r1[t - 1][1]][r0[j][0]] + dis_mat[r0[j][1]][r1[t][0]]

                    rs_copy_violation = rs_violation
                    if d0 > capacity:
                        rs_copy_violation -= d0 - max(capacity, d0 - demand_mat[r0[j][0]][r0[j][1]])

                    d1_copy = d1 + demand_mat[r0[j][0]][r0[j][1]]
                    if d1_copy > capacity:
                        rs_copy_violation += min(demand_mat[r0[j][0]][r0[j][1]], d1_copy - capacity)

                    r1.insert(t, r0[j])
                    r0.pop(j)
                    rs_copy_copy = copy.copy(rs_copy)
                    if len(r0) == 2:
                        rs_copy.pop(i)
                    yield rs_copy, rs_copy_cost, rs_copy_violation

                    rs_copy = rs_copy_copy
                    rs_copy[k] = copy.copy(rs_copy[k])
                    r0 = rs[i]
                    r1 = rs[k]

                    rs_copy_cost = rs_copy_cost - dis_mat[r1[t - 1][1]][r0[j][0]] - dis_mat[r0[j][1]][r1[t][0]] \
                                                + dis_mat[r1[t - 1][1]][r0[j][1]] + dis_mat[r0[j][0]][r1[t][0]]

                    rs_copy[k][t] = (rs_copy[k][t][1], rs_copy[k][t][0])
                    if len(rs[i]) == 3:
                        rs_copy.pop(i)
                    yield rs_copy, rs_copy_cost, rs_copy_violation

def double_insertion(rs, rs_cost, rs_violation, dis_mat):
    # double insertion in the same route

    rs_copy_violation = rs_violation
    for k in range(0, len(rs)):
        # i is the head
        if len(rs[k]) <= 4:
            continue
        for i in range(1, len(rs[k]) - 2):
            for j in range(1, len(rs[k])):
                if j == i or j == i + 1 or j == i + 2:
                    continue
                rs_copy = copy.copy(rs)
                rs_copy[k] = r = copy.copy(rs[k])
                rs_copy_cost = rs_cost - dis_mat[r[i - 1][1]][r[i][0]] - dis_mat[r[i + 1][1]][r[i + 2][0]] \
                                       + dis_mat[r[i - 1][1]][r[i + 2][0]] \
                                       - dis_mat[r[j - 1][1]][r[j][0]] \
                                       + dis_mat[r[j - 1][1]][r[i][0]] + dis_mat[r[i + 1][1]][r[j][0]]

                if i < j:
                    rs_copy[k] = r[:i] + r[i+2:j] + r[i:i+2] + r[j:]
                else:
                    rs_copy[k] = r[:j] + r[i:i+2] + r[j:i] + r[i+2:]
                yield rs_copy, rs_copy_cost, rs_copy_violation

                rs_copy = copy.copy(rs_copy)
                rs_copy[k] = copy.copy(rs_copy[k])
                r = rs[k]
                rs_copy_cost = rs_copy_cost - dis_mat[r[j - 1][1]][r[i][0]] - dis_mat[r[i + 1][1]][r[j][0]] \
                                            + dis_mat[r[j - 1][1]][r[i + 1][1]] + dis_mat[r[i][0]][r[j][0]]
                if i < j:
                    rs_copy[k][j - 1] = (r[i][1], r[i][0])
                    rs_copy[k][j - 2] = (r[i+1][1], r[i+1][0])
                else:
                    rs_copy[k][j + 1] = (r[i][1], r[i][0])
                    rs_copy[k][j] = (r[i+1][1], r[i+1][0])

                yield rs_copy, rs_copy_cost, rs_copy_violation

    # double insertion across routes
    for i in range(0, len(rs)):
        if len(rs[i]) <= 3:
            continue
        d0 = sum([demand_mat[n[0]][n[1]] for n in rs[i]])
        for k in range(0, len(rs)):
            if (i == k):
                continue
            d1 = sum([demand_mat[n[0]][n[1]] for n in rs[k]])

            # j is the head
            for j in range(1, len(rs[i]) - 2):
                # insert before t
                for t in range(1, len(rs[k])):
                    rs_copy = copy.copy(rs)
                    r0 = rs_copy[i] = copy.copy(rs[i])
                    r1 = rs_copy[k] = copy.copy(rs[k])

                    rs_copy_cost = rs_cost - dis_mat[r0[j - 1][1]][r0[j][0]] - dis_mat[r0[j + 1][1]][r0[j + 2][0]] \
                                           - dis_mat[r1[t - 1][1]][r1[t][0]] \
                                           + dis_mat[r0[j - 1][1]][r0[j + 2][0]] \
                                           + dis_mat[r1[t - 1][1]][r0[j][0]] + dis_mat[r0[j + 1][1]][r1[t][0]]

                    rs_copy_violation = rs_violation
                    if d0 > capacity:
                        rs_copy_violation -= d0 - max(capacity, d0 - demand_mat[r0[j][0]][r0[j][1]] - demand_mat[r0[j+1][0]][r0[j+1][1]])

                    d1_copy = d1 + demand_mat[r0[j][0]][r0[j][1]] + demand_mat[r0[j+1][0]][r0[j+1][1]]
                    if d1_copy > capacity:
                        rs_copy_violation += min(demand_mat[r0[j][0]][r0[j][1]] + demand_mat[r0[j+1][0]][r0[j+1][1]], d1_copy - capacity)

                    rs_copy[i] = r0[:j] + r0[j+2:]
                    rs_copy[k] = r1[:t] + r0[j:j+2] + r1[t:]

                    rs_copy_copy = copy.copy(rs_copy)
                    if len(rs_copy[i]) == 2:
                        rs_copy.pop(i)
                    yield rs_copy, rs_copy_cost, rs_copy_violation

                    rs_copy = rs_copy_copy
                    rs_copy[k] = copy.copy(rs_copy[k])
                    r0 = rs[i]
                    r1 = rs[k]
                    rs_copy_cost = rs_copy_cost - dis_mat[r1[t - 1][1]][r0[j][0]] - dis_mat[r0[j + 1][1]][r1[t][0]] \
                                                + dis_mat[r1[t - 1][1]][r0[j + 1][1]] + dis_mat[r0[j][0]][r1[t][0]]
                    rs_copy[k][t] = (r0[j + 1][1], r0[j + 1][0])
                    rs_copy[k][t + 1] = (r0[j][1], r0[j][0])

                    if len(rs[i]) == 4:
                        rs_copy.pop(i)

                    yield rs_copy, rs_copy_cost, rs_copy_violation


def swap(rs, rs_cost, rs_violation, dis_mat):
    # swap within a route
    for k in range(0, len(rs)):
        r = rs[k]
        for i in range(1, len(r)-1):
            for j in range(i+2, len(r)-1):
                rs_copy = copy.copy(rs)
                rs_copy[k] = copy.copy(rs_copy[k])
                rs_copy_cost = rs_cost - dis_mat[r[i-1][1]][r[i][0]] - dis_mat[r[i][1]][r[i+1][0]] \
                                       - dis_mat[r[j-1][1]][r[j][0]] - dis_mat[r[j][1]][r[j+1][0]] \
                                       + dis_mat[r[i-1][1]][r[j][0]] + dis_mat[r[j][1]][r[i+1][0]] \
                                       + dis_mat[r[j-1][1]][r[i][0]] + dis_mat[r[i][1]][r[j+1][0]]
                rs_copy_violation = rs_violation
                rs_copy[k][i], rs_copy[k][j] = r[j], r[i]
                yield rs_copy, rs_copy_cost, rs_copy_violation

                rs_copy_ori = rs_copy
                rs_copy_cost_ori = rs_copy_cost

                rs_copy = copy.copy(rs_copy)
                rs_copy[k] = copy.copy(rs_copy[k])
                rs_copy_cost = rs_copy_cost_ori - dis_mat[r[i-1][1]][r[j][0]] - dis_mat[r[j][1]][r[i+1][0]] \
                                       + dis_mat[r[i-1][1]][r[j][1]] + dis_mat[r[j][0]][r[i+1][0]]
                rs_copy[k][i] = (r[j][1], r[j][0])
                yield rs_copy, rs_copy_cost, rs_copy_violation

                rs_copy = copy.copy(rs_copy_ori)
                rs_copy[k] = copy.copy(rs_copy[k])
                rs_copy_cost = rs_copy_cost_ori - dis_mat[r[j-1][1]][r[i][0]] - dis_mat[r[i][1]][r[j+1][0]] \
                                       + dis_mat[r[j-1][1]][r[i][1]] + dis_mat[r[i][0]][r[j+1][0]]
                rs_copy[k][j] = (r[i][1], r[i][0])
                yield rs_copy, rs_copy_cost, rs_copy_violation

                rs_copy = copy.copy(rs_copy)
                rs_copy[k] = copy.copy(rs_copy[k])
                rs_copy_cost = rs_copy_cost - dis_mat[r[i-1][1]][r[j][0]] - dis_mat[r[j][1]][r[i+1][0]] \
                                            + dis_mat[r[i-1][1]][r[j][1]] + dis_mat[r[j][0]][r[i+1][0]]
                rs_copy[k][i] = (r[j][1], r[j][0])
                yield rs_copy, rs_copy_cost, rs_copy_violation


    # swap across routes
    for i in range(0, len(rs)):
        d0 = sum([demand_mat[n[0]][n[1]] for n in rs[i]])
        r0 = rs[i]
        for j in range(i+1, len(rs)):
            d1 = sum([demand_mat[n[0]][n[1]] for n in rs[j]])
            r1 = rs[j]
            for k in range(1, len(r0)-1):
                for t in range(1, len(r1)-1):
                    rs_copy = copy.copy(rs)
                    rs_copy[i] = copy.copy(r0)
                    rs_copy[j] = copy.copy(r1)
                    rs_copy_cost = rs_cost - dis_mat[r0[k - 1][1]][r0[k][0]] - dis_mat[r0[k][1]][r0[k + 1][0]] \
                                           - dis_mat[r1[t - 1][1]][r1[t][0]] - dis_mat[r1[t][1]][r1[t + 1][0]] \
                                           + dis_mat[r0[k - 1][1]][r1[t][0]] + dis_mat[r1[t][1]][r0[k + 1][0]] \
                                           + dis_mat[r1[t - 1][1]][r0[k][0]] + dis_mat[r0[k][1]][r1[t + 1][0]]
                    rs_copy_violation = rs_violation - max(0, d0 - capacity) + max(0, d0 - demand_mat[r0[k][0]][r0[k][1]] + demand_mat[r1[t][0]][r1[t][1]] - capacity) \
                                                     - max(0, d1 - capacity) + max(0, d1 - demand_mat[r1[t][0]][r1[t][1]] + demand_mat[r0[k][0]][r0[k][1]] - capacity)
                    rs_copy[i][k], rs_copy[j][t] = rs_copy[j][t], rs_copy[i][k]
                    yield rs_copy, rs_copy_cost, rs_copy_violation

                    rs_copy_ori = rs_copy
                    rs_copy_cost_ori = rs_copy_cost

                    rs_copy = copy.copy(rs_copy)
                    rs_copy[i] = copy.copy(rs_copy[i])
                    rs_copy_cost = rs_copy_cost_ori - dis_mat[r0[k - 1][1]][r1[t][0]] - dis_mat[r1[t][1]][r0[k + 1][0]] \
                                                    + dis_mat[r0[k - 1][1]][r1[t][1]] + dis_mat[r1[t][0]][r0[k + 1][0]]
                    rs_copy[i][k] = (rs_copy[i][k][1], rs_copy[i][k][0])
                    yield rs_copy, rs_copy_cost, rs_copy_violation

                    rs_copy = copy.copy(rs_copy_ori)
                    rs_copy[j] = copy.copy(rs_copy_ori[j])
                    rs_copy_cost = rs_copy_cost_ori - dis_mat[r1[t - 1][1]][r0[k][0]] - dis_mat[r0[k][1]][r1[t + 1][0]] \
                                                    + dis_mat[r1[t - 1][1]][r0[k][1]] + dis_mat[r0[k][0]][r1[t + 1][0]]
                    rs_copy[j][t] = (rs_copy[j][t][1], rs_copy[j][t][0])
                    yield rs_copy, rs_copy_cost, rs_copy_violation

                    rs_copy = copy.copy(rs_copy)
                    rs_copy[i] = copy.copy(rs_copy[i])
                    rs_copy_cost = rs_copy_cost - dis_mat[r0[k - 1][1]][r1[t][0]] - dis_mat[r1[t][1]][r0[k + 1][0]] \
                                   + dis_mat[r0[k - 1][1]][r1[t][1]] + dis_mat[r1[t][0]][r0[k + 1][0]]
                    rs_copy[i][k] = (rs_copy[i][k][1], rs_copy[i][k][0])
                    yield rs_copy, rs_copy_cost, rs_copy_violation


def crossover(sol1, sol2, dis_mat):
    rs = copy.deepcopy(sol1[0])
    a = random.randrange(0, len(sol1[0]))
    b = random.randrange(0, len(sol2[0]))
    rs[a] = copy.copy(sol2[0][b])
    task_pos = {}
    task_is_served = {}
    for i, r in enumerate(rs):
        for j in range(1, len(r)-1):
            task = rs[i][j]
            task_is_served[(task[0], task[1])] = True
            task_is_served[(task[1], task[0])] = True

            task_tf = (task[0], task[1]) if (task[0] < task[1]) else (task[1], task[0])
            if task_pos.get(task_tf):
                task_pos[task_tf].append((i, j))
            else:
                task_pos[task_tf] = [(i, j)]
    
    duplicated_tasks = [task for task in task_pos.keys() if len(task_pos[task]) > 1]
    unserved_tasks = []

    for d in demands:
        if not task_is_served.get((d[0], d[1])):
            # print(task_is_served.get((d[0], d[1])), task_is_served.get((d[1], d[0])))
            unserved_tasks.append(d)
            task_is_served[(d[0], d[1])] = task_is_served[(d[1], d[0])] = True

    # print(rs)
    # print(duplicated_tasks, unserved_tasks)
    rm = []
    for task in duplicated_tasks:
        task_tf = (task[0], task[1]) if (task[0] < task[1]) else (task[1], task[0])
        i0, j0 = task_pos[task_tf][0]
        r = rs[i0]
        delta0 = dis_mat[r[j0-1][1]][r[j0][0]] + dis_mat[r[j0][1]][r[j0+1][0]] - dis_mat[r[j0-1][1]][r[j0+1][0]]

        i1, j1 = task_pos[task_tf][1]
        r = rs[i1]
        delta1 = dis_mat[r[j1-1][1]][r[j1][0]] + dis_mat[r[j1][1]][r[j1+1][0]] - dis_mat[r[j1-1][1]][r[j1+1][0]]
        if (delta0 > delta1):
            rm.append((i0, j0))
        else:
            rm.append((i1, j1))

    rm.sort(reverse=True)
    for rmo in rm:
        rs[rmo[0]].pop(rmo[1])

    random.shuffle(unserved_tasks)
    for pac in unserved_tasks:
        task = (pac[0], pac[1])
        dem = pac[2]
        best_c = INT_INF
        best_pos = None
        best_tas = None

        for k, r in enumerate(rs):
            d = sum([demand_mat[n[0]][n[1]] for n in r])
            if d + dem > capacity:
                continue
            for i in range(1, len(r)):
                c = -dis_mat[r[i-1][1]][r[i][0]] + cost_mat[task[0]][task[1]] \
                    +dis_mat[r[i-1][1]][task[0]] + dis_mat[task[1]][r[i][0]]
                if (c < best_c):
                    best_pos = (k, i)
                    best_c = c
                    best_tas = task

                c = -dis_mat[r[i-1][1]][r[i][0]] + cost_mat[task[0]][task[1]] \
                    +dis_mat[r[i-1][1]][task[1]] + dis_mat[task[0]][r[i][0]] 
                if (c < best_c):
                    best_pos = (k, i)
                    best_c = c
                    best_tas = (task[1], task[0])
            
        if best_pos:
            rs[best_pos[0]].insert(best_pos[1], best_tas)
        else:
            rs.append([(depot, depot), task, (depot, depot)])

    rs = [r for r in rs if len(r) > 2]
    return rs


def cal_tc(solution):
    total_cost = 0
    for route in solution:
        path_end = depot
        for task in route:

            total_cost += dis_mat[path_end][task[0]] + cost_mat[task[0]][task[1]]
            path_end = task[1]
        total_cost += dis_mat[path_end][depot]

    return total_cost


def cal_tv(solution):
    total_violation = 0
    for route in solution:
        route_demand = sum([demand_mat[task[0]][task[1]] for task in route])
        total_violation += max(0, route_demand - capacity)
    return total_violation

def add_dummy_task(solution):
    for r in solution:
        r.append((depot, depot))
        r.insert(0, (depot, depot))

def evaluate(cost, violation, cost_best):
    lam = (cost_best / capacity) * (cost_best/cost + violation/capacity + 1)
    return cost + lam * violation

def valid_best_sol(results):
    valid_results = [result for result in results if result[2] == 0]
    valid_results.sort(key=lambda x: x[1])
    return valid_results[0]

def eva_best_sol(results):
    results.sort(key=lambda result: result[3])
    return results[0]

def ps(cost_mat, dis_mat, demands, timeout, results):
    st = time.time()
    optimal_rs, min_total_cost = path_scanning(cost_mat, dis_mat, demands)
    round_time = time.time() - st

    time_left = timeout - round_time
    
    while time_left > round_time:
        st = time.time()
        rs, total_cost = path_scanning(cost_mat, dis_mat, demands)
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            optimal_rs = rs

        rt = time.time() - st
        time_left -= rt
        round_time = round_time * 0.6 + rt * 0.4
    
    add_dummy_task(optimal_rs)
    results.append((optimal_rs, min_total_cost, 0))

def local_search(sol, best_cost):
    rs, rs_cost, rs_violation, rs_eva = sol
    sol_best = sol
    for sol_rsc in single_insertion(rs, rs_cost, rs_violation, dis_mat):
        rsc, rsc_cost, rsc_violation = sol_rsc
        rsc_eva = evaluate(rsc_cost, rsc_violation, best_cost)
        if rsc_eva < sol_best[3]:
            sol_best = sol_rsc + (rsc_eva,)

    for sol_rsc in double_insertion(rs, rs_cost, rs_violation, dis_mat):
        rsc, rsc_cost, rsc_violation = sol_rsc
        rsc_eva = evaluate(rsc_cost, rsc_violation, best_cost)
        if rsc_eva < sol_best[3]:
            sol_best = sol_rsc + (rsc_eva,)


    for sol_rsc in swap(rs, rs_cost, rs_violation, dis_mat):
        rsc, rsc_cost, rsc_violation = sol_rsc
        rsc_eva = evaluate(rsc_cost, rsc_violation, best_cost)
        if rsc_eva < sol_best[3]:
            sol_best = sol_rsc + (rsc_eva,)
    return sol_best


def crossover_search(results, dis_mat, cost_best):
    a = random.randrange(0, len(results))
    b = random.randrange(0, len(results))
    if len(results) != 1:
        while a == b:
            b = random.randrange(0, len(results))

    rs = crossover(results[a], results[b], dis_mat)
    tcc = cal_tc(rs)
    sol = (rs, tcc, 0, evaluate(tcc, 0, cost_best))
    # if random.random() < .9:
    #     return local_search(sol, cost_best)
    # else:
    #     return sol
    return local_search(sol, cost_best)

def f3(x):
    return x[3]

if __name__ == '__main__':
    dis_mat = minimal_distance(cost_mat)

    max_worker = 8
    pool = Pool(max_worker)
    manager = multiprocessing.Manager()
    results = manager.list()
    timeout = (args.t - (time.time() - st)) / 2

    pool.starmap(ps, [[cost_mat, dis_mat, demands, timeout, results] for _ in range(max_worker)])

    best_cost = min([so[1] for so in results])
    for i in range(0, len(results)):
        results[i] = results[i] + (evaluate(results[i][1], results[i][2], best_cost),)


    t0 = time.time()
    results_co = pool.starmap(crossover_search, [[results, dis_mat, best_cost] for sol in results])
    results += results_co
    results.sort(key=f3)
    results = results[:max_worker]
    best_cost = min([so[1] for so in results])

    real_time_spent = time.time() - t0
    est_time_spent = real_time_spent

    time_left = args.t - (time.time() - st) - 2
    rounds = 1
    while time_left > est_time_spent:
        rounds += 1
        t0 = time.time()
        results_co = pool.starmap(crossover_search, [[results, dis_mat, best_cost] for sol in results])
        results += results_co
        results.sort(key=f3)
        results = results[:max_worker]

        best_cost = min([so[1] for so in results])
        real_time_spent = time.time() - t0
        est_time_spent = est_time_spent * 0.6 + real_time_spent * 0.4
        time_left -= real_time_spent

    best_sol = valid_best_sol(results)
    rs = [r[1:-1] for r in best_sol[0]]
    output(rs, best_sol[1])
    print(rounds)
    print(f'Time left {args.t - (time.time() - st)}')
    
