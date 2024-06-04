import numpy as np
import itertools
import random
import torch

def solveByBackTrack(nums):
    def backtrack(position, end):

        if position == end:
            res.append(nums[:])
            return

        for index in range(position, end):
            nums[index], nums[position] = nums[position], nums[index]
            backtrack(position + 1, end)
            nums[index], nums[position] = nums[position], nums[index]

    res = []
    backtrack(0, len(nums))
    return res


def getArrangeByBackTrack(n, k):
    nums = list(range(1, n + 1))
    # nums = [3,2,1,4,5]
    candidates = set()  # 去重
    results = solveByBackTrack(nums)  # 全排列
    print(results[0][0])

    for i in results:
        tmp = i[:k]  # 截断
        tmp.sort()
        candidates.add(','.join(list(map(lambda x: str(x), tmp))))
        candidates_list = list(candidates)
        candidates_list.sort()
    return candidates_list


# 输入list，返回全排列list，二维list
def getArrangement(nums):
    nums.sort()
    arrangement_tuple = list(itertools.permutations(nums, len(nums)))
    arrangement_list = []

    for i in arrangement_tuple:
        arrangement_list.append(list(i))
    return arrangement_list


# 求nums的组合，二维数组
def getCombination(nums, k):
    combination_tuple = list(itertools.combinations(nums, k))
    combination_list = []

    for i in combination_tuple:
        combination_list.append(list(i))
    return combination_list


# 求k=1到n的所有组合，返回三维数组
def getAllCombination(nums):
    nums.sort()
    all_combination_list = []

    for k in range(0, len(nums)):
        all_combination_list.append(getCombination(nums, k + 1))
    # print(all_combination_list)
    return all_combination_list



# 输入全部组合表，返回utility函数值的表，二维数组，与组合表对应
def getAllCombinationUtilityValue(args, all_combination_list, nets, train_data, client,num):

    ini_net = nets[client]
    sum_w = ini_net.state_dict()
    all_combination_utility_value_list = []

    for i in range(0, len(all_combination_list)):
        all_combination_utility_value_list.append([])

        for j in range(0, len(all_combination_list[i])):

            correct, total = 0, 0
            for _, indice in enumerate(all_combination_list[i][j]):
                net_para = nets[indice].state_dict()
                if _ == 0:
                    for key in net_para:
                        sum_w[key] = net_para[key] * 1/len(all_combination_list[i][j])
                else:
                    for key in net_para:
                        sum_w[key] += net_para[key] * 1/len(all_combination_list[i][j])

            ini_net.load_state_dict(sum_w)          # Update the global model
            ini_net.to(args.device)
            ini_net.eval()

            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(args.device, dtype=torch.float32), target.to(args.device, dtype=torch.int64)
                out = ini_net(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
            all_combination_utility_value_list[i].append(correct / float(total))


    # print(all_combination_utility_value_list)
    return all_combination_utility_value_list


def getSharplyValue(args, nums, client, nets, train_data):
    # 判断client是不是在数组中，不在就直接返回报错
    if client not in nums:
        print("Client j is not in the set.")
        return

    # 根据输入集合，获得全排列的表
    arrangement_list = getArrangement(nums)

    # 根据输入集合，获得所有组合的表
    # Cnk, n = len(nums), k = 1...n
    all_combination_list = getAllCombination(nums)

    # 获得utility_value表
    # TODO:更新utility_value表
    all_combination_utility_value_list = getAllCombinationUtilityValue(args, all_combination_list, nets, train_data, client, len(nums))
    # all_combination_utility_value_list = [[100,125,50],[270,375,350],[500]]

    # 初始化sharply value
    sharply_value_client = 0
    print('all_combination_utility_value_list = {}'.format(all_combination_utility_value_list))
    print('all_combination_list = {}'.format(all_combination_list))

    dictionary_shapley = {}

    # 遍历所有的全排列
    for clint in nums:
        for i in range(0, len(arrangement_list)):

            # 获取client j 在这一次排列中的下标index_client
            index_client = 0
            for j in range(0, len(arrangement_list[i])):
                if client == arrangement_list[i][j]:
                    index_client = j
                    break

            # 获取这一次排列的没有client的组合,然后排序可以和组合表相比较
            once_permutation_without_client_list = arrangement_list[i][0:index_client]
            once_permutation_without_client_list.sort()
            # print(arrangement_list[i])
            # print(once_permutation_without_client_list)

            # 获取这一次排列的有client的组合，然后排序可以和组合表相比较
            once_permutation_with_client_list = arrangement_list[i][0:index_client + 1]
            once_permutation_with_client_list.sort()
            # print(arrangement_list[i])
            # print(once_permutation_with_client_list)
            # print(index_client)

            # 当client 是这一次排列的第一个数，则可以直接加上这一次排列的sharply值
            # V set(client)-V set(null) = utility_value_with_client - 0, 空集对应的值是0
            if index_client == 0:
                client_list_temp = [client]
                for index_combination in range(0, len(all_combination_list[0])):

                    if (client_list_temp == all_combination_list[0][index_combination]):
                        utility_value_with_client = all_combination_utility_value_list[0][index_combination]

                # set(client)-set(null) = utility_value_with_client - 0, 空集对应的值是0
                sharply_value_once_permutation = utility_value_with_client - 0
                sharply_value_client = sharply_value_client + sharply_value_once_permutation
                # print(sharply_value_once_permutation)



            # 当client不是这一次排列的第一个数，减去client的集合不是空集的时候
            # V set(with_client)-V set(without_client) = utility_value_with_client - utility_value_without_client
            else:
                # 这一次排列的有client的组合的长度
                once_permutation_with_client_list_length = len(once_permutation_with_client_list)
                # 这一次排列的没有client的组合的长度
                once_permutation_without_client_list_length = len(once_permutation_without_client_list)

                # 求有client和没有clienr的utility值,将截取后的数组与总组合数组比较找到下标,再用下标在utility value表找值
                utility_value_with_client = 0
                utility_value_without_client = 0

                # 找有client的utility_value
                for index_combination in range(0, len(all_combination_list[once_permutation_with_client_list_length - 1])):
                    if (once_permutation_with_client_list ==
                            all_combination_list[once_permutation_with_client_list_length - 1][index_combination]):
                        utility_value_with_client = \
                        all_combination_utility_value_list[once_permutation_with_client_list_length - 1][index_combination]
                        break

                # 找没有client的utility_value
                for index_combination in range(0, len(arrangement_list[once_permutation_without_client_list_length - 1])):
                    if (once_permutation_without_client_list ==
                            all_combination_list[once_permutation_without_client_list_length - 1][index_combination]):
                        utility_value_without_client = \
                        all_combination_utility_value_list[once_permutation_without_client_list_length - 1][
                            index_combination]
                        break

                # 求本次排列的sharply值，并且加到shaply value中
                sharply_value_once_permutation = utility_value_with_client - utility_value_without_client
                sharply_value_client = sharply_value_client + sharply_value_once_permutation
                # print(sharply_value_once_permutation)
            # print(sharply_value_client)
            # print()

        # 将每一次排列求的值除以排列数量得到最后的sharply value
        sharply_value_client = sharply_value_client / len(arrangement_list)
        # print('sharply_value_client = {}'.format(sharply_value_client))
        dictionary_shapley[clint] = sharply_value_client
    return dictionary_shapley


def getSharplyValueWithMonteSampling(nums, client):
    # 判断client是不是在数组中，不在就直接返回报错
    if client not in nums:
        print("Client j is not in the set.")
        return

    # 根据输入集合，获得全排列的表
    arrangement_list = getArrangement(nums)

    # 根据输入集合，获得所有组合的表
    # Cnk, n = len(nums), k = 1...n
    all_combination_list = getAllCombination(nums)

    # 获得utility_value表
    # TODO:更新utility_value表
    all_combination_utility_value_list = getAllCombinationUtilityValue(all_combination_list)
    # all_combination_utility_value_list = [[100, 125, 50], [270, 375, 350], [500]]

    # 初始化sharply value
    sharply_value_client = 0
    # print(all_combination_utility_value_list)
    # print(all_combination_list)

    # 遍历所有的全排列
    for i in range(0, len(arrangement_list)):

        # 获取client j 在这一次排列中的下标index_client
        index_client = 0
        for j in range(0, len(arrangement_list[i])):
            if client == arrangement_list[i][j]:
                index_client = j
                break

        # 获取这一次排列的没有client的组合,然后排序可以和组合表相比较
        once_permutation_without_client_list = arrangement_list[i][0:index_client]
        once_permutation_without_client_list.sort()
        # print(arrangement_list[i])
        # print(once_permutation_without_client_list)

        # 获取这一次排列的有client的组合，然后排序可以和组合表相比较
        once_permutation_with_client_list = arrangement_list[i][0:index_client + 1]
        once_permutation_with_client_list.sort()
        # print(arrangement_list[i])
        # print(once_permutation_with_client_list)
        # print(index_client)

        # 当client 是这一次排列的第一个数，则可以直接加上这一次排列的sharply值
        # V set(client)-V set(null) = utility_value_with_client - 0, 空集对应的值是0
        if index_client == 0:
            client_list_temp = [client]
            for index_combination in range(0, len(all_combination_list[0])):

                if (client_list_temp == all_combination_list[0][index_combination]):
                    utility_value_with_client = all_combination_utility_value_list[0][index_combination]

            # set(client)-set(null) = utility_value_with_client - 0, 空集对应的值是0
            sharply_value_once_permutation = utility_value_with_client - 0
            sharply_value_client = sharply_value_client + sharply_value_once_permutation



        # 当client不是这一次排列的第一个数，减去client的集合不是空集的时候
        # V set(with_client)-V set(without_client) = utility_value_with_client - utility_value_without_client
        else:
            # 这一次排列的有client的组合的长度
            once_permutation_with_client_list_length = len(once_permutation_with_client_list)
            # 这一次排列的没有client的组合的长度
            once_permutation_without_client_list_length = len(once_permutation_without_client_list)

            # 求有client和没有clienr的utility值,将截取后的数组与总组合数组比较找到下标,再用下标在utility value表找值
            utility_value_with_client = 0
            utility_value_without_client = 0

            # 找有client的utility_value
            for index_combination in range(0, len(all_combination_list[once_permutation_with_client_list_length - 1])):
                if (once_permutation_with_client_list ==
                        all_combination_list[once_permutation_with_client_list_length - 1][index_combination]):
                    utility_value_with_client = \
                    all_combination_utility_value_list[once_permutation_with_client_list_length - 1][index_combination]
                    break

            # 找没有client的utility_value
            for index_combination in range(0, len(arrangement_list[once_permutation_without_client_list_length - 1])):
                if (once_permutation_without_client_list ==
                        all_combination_list[once_permutation_without_client_list_length - 1][index_combination]):
                    utility_value_without_client = \
                    all_combination_utility_value_list[once_permutation_without_client_list_length - 1][
                        index_combination]
                    break

            # 求本次排列的sharply值，并且加到shaply value中
            sharply_value_once_permutation = utility_value_with_client - utility_value_without_client
            sharply_value_client = sharply_value_client + sharply_value_once_permutation
        print((sharply_value_once_permutation))
        print(sharply_value_client)
        print()

    # 将每一次排列求的值除以排列数量得到最后的sharply value
    sharply_value_client = sharply_value_client / len(arrangement_list)
    print(len(arrangement_list))
    return sharply_value_client


if __name__ == "__main__":
    # 全排列

    # clients
    nums = [1, 2, 3]
    # re = getArrangement(nums)
    # tem = getAllCombination(nums)
    # tem2 = getAllCombinationUtilityValue(tem)
    q1 = [1, 2]
    q2 = [2, 1]
    # print(q1 == q2)
    print('client set = {}'.format(nums))
    print()
    print("client 1 SV:")
    re1 = getSharplyValue(nums, 1)
    print()
    print("client 2 SV:")
    re2 = getSharplyValue(nums, 2)
    print()
    print("client 3 SV:")
    re3 = getSharplyValue(nums, 3)
    print()
    print('total SV = {}'.format(re1 + re2 + re3))
