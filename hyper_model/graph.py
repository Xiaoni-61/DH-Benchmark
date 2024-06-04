import copy

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot  as plt


def Generate_competetive_graph(Compet_ratio, usr_num):
    Compet_Matrix = np.zeros([usr_num,usr_num])
    for i in range(usr_num):
        for j in range(i+1,usr_num):
            p = random.random()
            if p< Compet_ratio:
                Compet_Matrix[i,j]=1
                Compet_Matrix[j,i]=1

    return Compet_Matrix

def generate_coalition_group(Compet_Matrix):
    negative_compete_matrix = np.zeros([len(Compet_Matrix),len(Compet_Matrix)])
    coalition_group = []

    for i in range(len(Compet_Matrix)):
        for j in range(len(Compet_Matrix)):
            if(Compet_Matrix[i,j]==1):
                negative_compete_matrix[i,j]= 0
            elif(Compet_Matrix[i,j]==0):
                negative_compete_matrix[i,j]= 1

    negative_compete_graph = nx.from_numpy_matrix(negative_compete_matrix)
    negative_compete_graph_tmp = copy.deepcopy(negative_compete_graph)

    cliques = nx.find_cliques(negative_compete_graph_tmp)
    cliques_list = list(cliques)
    while len(cliques_list) > 0:
        cliques_list.sort(key=len,reverse=True)
        coalition_group.append(cliques_list[0])
        [negative_compete_graph_tmp.remove_node(nd) for nd in cliques_list[0]]
        cliques_list = list(nx.find_cliques(negative_compete_graph_tmp))

    return coalition_group


def generate_Sequence(usr_num,Benefit_Matrix):

    Contribution = np.sum((np.ones([usr_num,usr_num])-np.eye(usr_num))*Benefit_Matrix,axis=0)
    S = np.argsort(-Contribution)
    return S

def ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix, Compet_Matrix):
    B_i_value = []
    B_i_tmp = []
    vi = usr_i
    Contribute_Matrix_tmp = copy.deepcopy(Contribute_Matrix)
    G_u = nx.from_numpy_matrix(Contribute_Matrix_tmp, create_using=nx.DiGraph)



    for j in range(len(Benefit_Matrix[vi])):
        if(Benefit_Matrix[vi][j] > 0 and j != vi):
            B_i_tmp.append(j)
            B_i_value.append(Benefit_Matrix[vi][j])
    B_i_value_arr = np.array(B_i_value)
    B_i_tmp_arr = np.array(B_i_tmp)
    B_i = copy.deepcopy(B_i_tmp_arr[np.argsort(-B_i_value_arr)])


    for vj in B_i:
        V_j = []
        S_j = []
        S_ij_1 = []
        V_i = []
        S_i = []
        S_ij_2 = []

        G_u = nx.from_numpy_matrix(Contribute_Matrix_tmp, create_using=nx.DiGraph)
        for usage_to_vj in range(usr_num):
            if nx.has_path(G_u,usage_to_vj,vj):
                V_j.append(usage_to_vj)

        for client in V_j:
            for idx in range(len(Compet_Matrix[client])):
                if  Compet_Matrix[client][idx] == 1 and idx not in S_j:
                    S_j.append(idx)

        for client in S_j:
            if nx.has_path(G_u,vi,client):
                S_ij_1.append(client)

        for usage_from_vi in range(usr_num):
            if nx.has_path(G_u, vi, usage_from_vi):
                V_i.append(usage_from_vi)

        for client in V_i:
            for idx in range(len(Compet_Matrix[client])):
                if Compet_Matrix[client][idx] == 1 and idx not in S_i:
                    S_i.append(idx)

        for client in S_i:
            if nx.has_path(G_u,client,vj):
                S_ij_2.append(client)


        if len(S_ij_1) == 0 and len(S_ij_2) == 0:
            Contribute_Matrix_tmp[vj,vi] = 1
        else:
            Contribute_Matrix_tmp[vj,vi] = 0

    return Contribute_Matrix_tmp




def Greedy_allocation(usr_num,Compet_Matrix,Benefit_Matrix):
    Sequence = generate_Sequence(usr_num,Benefit_Matrix)
    Contribute_Matrix = np.diag([1]*usr_num)

    for usr_i in Sequence:
        Contribute_Matrix = ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix, Compet_Matrix)

    Contribute_Matrix = Contribute_Matrix.T
    return Contribute_Matrix






