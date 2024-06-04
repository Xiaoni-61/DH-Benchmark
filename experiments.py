import json
import heapq
import sys
from itertools import combinations

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import time
from tqdm import tqdm

from sharplyValue import getSharplyValue
import random

import datetime
# from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from torch.nn.functional import kl_div
from collections import Counter
import cvxpy as cp
from hyper_model.benefit import Training_all
import seaborn as sns
import matplotlib.pyplot as plt


# import seaborn as sns
# import matplotlib.pyplot as plt


class UnbufferedFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        # Open the file with buffering set to 0 (unbuffered)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay, errors=None)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=0,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=1, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--k', type=int, default=2, help='pFedSV k')
    parser.add_argument('--lamda', type=float, default=0.5, help='pFedJS lambda')
    parser.add_argument('--round_calculate', type=int, default=5, help='every round calculate acc')

    parser.add_argument('--difference_measure', type=str, default='all',
                        help='pFedgraph: how to measure the model difference')
    parser.add_argument('--alpha', type=float, default=0.8, help='pFedgraph Hyper-parameter to avoid concentration')
    parser.add_argument('--lam', type=float, default=0.01, help="pFedgraph Hyper-parameter in the objective")
    # RACE
    parser.add_argument('--K', type=int, default=3, help='RACE K: how many clients attend to this round')

    parser.add_argument('--train_pt', type=bool, default=True, help='CE')

    parser.add_argument('--SCFL_data_num', action='store_true',default=False, help='In SCFL if use data num to get weight vector')

    parser.add_argument('--use_feature', action='store_true',default=False,
                        help='in pFedJS, if use label or feature to calculate JS matrix')

    args = parser.parse_args()

    # pFedCollab in/out-dimension of model
    shapes_in = {
        'mnist': (1, 28, 28),
        'fmnist': (1, 28, 28),
        'femnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
        'rotated-cifar10': (3, 32, 32),
        'cifar100': (3, 32, 32),
        'coarse-cifar100': (3, 32, 32),
        'svhn': (3, 32, 32),
        'a9a': (1, 123),
        'rcv1': (1, 47236),
        'covtype': (1, 54),
    }

    shapes_out = {
        'mnist': 10,
        'fmnist': 10,
        'femnist': 10,
        'cifar10': 10,
        'rotated-cifar10': 10,
        'cifar100': 100,
        'coarse-cifar100': 20,
        'svhn': 10,
        'a9a': 2,
        'rcv1': 2,
        'covtype': 2,
    }

    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)

    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model + add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, data_distribution, train_dataloader, val_dataloader, test_dl, epochs, lr, args_optimizer,
              device="cpu"):
    logger.info('Training network %s' % str(net_id))
    net = net.to(device)
    net.to(device)
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # val_acc, conf_matrix = compute_accuracy(net, val_dataloader, get_confusion_matrix=True, device=device)
    # personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,
    #                                                                           device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training val accuracy: {}'.format(val_acc))
    # logger.info('>> Pre-Training personalized_test accuracy: {}'.format(personalized_test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                net.to(device)
                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        # logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # train_acc = compute_accuracy(net, train_dataloader, device=device)
        # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        # writer.add_scalar('Accuracy/train', train_acc, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # val_acc, conf_matrix = compute_accuracy(net, val_dataloader, get_confusion_matrix=True, device=device)
    # personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,
    #                                                                           device=device)

    # logger.info('>> clint %s, Training accuracy: %f' % (net_id, train_acc))
    # logger.info('>> clint %s, Test accuracy: %f' % (net_id, val_acc))
    # logger.info('>> clint %s, personalized_test accuracy: %f test_acc: %f' % (net_id, personalized_test_acc,generalized_test_acc))
    # print('>> clint {}, Training accuracy: {}'.format(net_id, train_acc))
    # print('>> clint {}, Val accuracy: {}'.format(net_id, val_acc))
    # print('>> clint {}, personalized_test accuracy: {} test_acc: {}'.format(net_id, personalized_test_acc, generalized_test_acc))
    print('>> clint {}'.format(net_id))

    net.to('cpu')
    logger.info(' ** clint %s Training complete **' % net_id)

    return


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                       args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                          weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args,
                   round, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2 - pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def find_largest_k_indices_excluding_idx(lst, k, n):
    # 创建一个新列表，包括所有元素的索引和值，但排除第n个元素
    elements_with_indices = [(index, value) for index, value in enumerate(lst) if index != n]

    # 使用heapq.nlargest找到最大的k个元素及其原始索引，按元素值排序
    largest_k = heapq.nlargest(k, elements_with_indices, key=lambda x: x[1])
    return [index for index, value in largest_k]


def local_train_FedCollab_net(nets_this_round, selected, args, train_dl_global, val_dl_global, relevance_vectors,
                              test_dl=None, device="cpu", ):
    nets = {}

    for net_single in nets_this_round:
        nets_this_round[net_single].to(device)
    for idx in range(args.n_parties):
        relevance_vector = relevance_vectors[idx]
        # for net_id, net in nets_this_round.items():
        train_local_dl = train_dl_global[idx]
        net = nets_this_round[idx]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                  weight_decay=args.reg)
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        net.train()
        if type(train_local_dl) == type([1]):
            pass
        else:
            train_local_dl = [train_local_dl]

        # for iteration in range(args.num_local_iterations):
        n_epoch = args.epochs
        for epoch in range(n_epoch):
            epoch_loss_collector = []
            for tmp in train_local_dl:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    loss = criterion(out, target)
                    # print("loss {}".format(loss))
                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            # logger.info('Client: %d, Epoch: %d Loss: %f' % (idx + 1, epoch, epoch_loss))
            # print("Client: {}， Epoch: {} Loss: {}".format(idx + 1, epoch, epoch_loss))

    for idx in range(args.n_parties):
        relevance_vector = relevance_vectors[idx]
        ini_net = nets_this_round[idx]
        sum_w = ini_net.state_dict()
        for idd, indice in enumerate(relevance_vector):
            net_para = nets_this_round[idd].state_dict()
            if idd == 0:
                for key in net_para:
                    sum_w[key] = net_para[key] * relevance_vector[idd]
            else:
                for key in net_para:
                    sum_w[key] += net_para[key] * relevance_vector[idd]
        ini_net.load_state_dict(sum_w)
        nets[idx] = ini_net

    return nets


def local_train_pFedSV_net(nets_this_round, selected, args, train_dl_global, val_dl_global, relevance_vectors,
                           test_dl=None, device="cpu", ):
    nets = {}
    a = 0.2

    weight_matrix = []
    for i in range(args.n_parties):
        weight_matrix.append([0 for p in range(args.n_parties)])

    for net_single in nets_this_round:
        nets_this_round[net_single].to(device)
    for idx in range(args.n_parties):
        # for net_id, net in nets_this_round.items():
        train_local_dl = train_dl_global[idx]
        net = nets_this_round[idx]
        # # Model Aggregation
        # for net_id, net in enumerate(nets_this_round.values()):
        #     net_para = net.state_dict()
        #     if net_id == 0:
        #         for key in net_para:
        #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        #     else:
        #         for key in net_para:
        #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]
        #

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                  weight_decay=args.reg)
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        net.train()
        if type(train_local_dl) == type([1]):
            pass
        else:
            train_local_dl = [train_local_dl]

        # for iteration in range(args.num_local_iterations):
        n_epoch = args.epochs
        for epoch in range(n_epoch):
            epoch_loss_collector = []
            for tmp in train_local_dl:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    loss = criterion(out, target)
                    # print("loss {}".format(loss))
                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            # logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            # print("Epoch: {} Loss: {}".format(epoch, epoch_loss))

    start_time = time.time()

    for idx in range(args.n_parties):
        train_local_dl = train_dl_global[idx]
        relevance_vector = relevance_vectors[idx]
        findVectorsindices = find_largest_k_indices_excluding_idx(relevance_vector, args.k, idx)  # Algorithm 1 7 line
        dictionary_shapley = {}
        w = {}
        findVectorsindices.append(idx)

        dictionary_shapley = getSharplyValue(args, copy.deepcopy(findVectorsindices), 0, nets_this_round,
                                             train_local_dl)  # Algorithm 1 8 line
        if dictionary_shapley is not None:
            for j, indice in enumerate(dictionary_shapley):
                relevance_vectors[idx][indice] = a * relevance_vectors[idx][indice] + (
                        1 - a) * dictionary_shapley[indice]
                weight_matrix[idx][indice] = max(dictionary_shapley[indice], 0)

    for idx in range(len(weight_matrix)):
        if sum(weight_matrix[idx]) == 0:
            weight_matrix[idx][idx] = 1
        weight_matrix[idx] = [x / sum(weight_matrix[idx]) for x in weight_matrix[idx]]

    for idx in range(args.n_parties):
        weight_vector = weight_matrix[idx]
        ini_net = nets_this_round[idx]
        sum_w = ini_net.state_dict()
        for idd, indice in enumerate(weight_vector):
            net_para = nets_this_round[idd].state_dict()
            if idd == 0:
                for key in net_para:
                    sum_w[key] = net_para[key] * weight_vector[idd]
            else:
                for key in net_para:
                    sum_w[key] += net_para[key] * weight_vector[idd]
        ini_net.load_state_dict(sum_w)
        nets[idx] = ini_net
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")

    return nets
    pass


def local_train_net(nets, selected, args, data_distributions, train_dl_global, val_dl_global, test_dl=None,
                    device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        data_distribution = data_distributions[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0
        #
        # if args.noise_type == 'space':
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
        #                                                          dataidxs, noise_level, net_id, args.n_parties - 1)
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
        #                                                          dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        train_net(net_id, net, data_distribution, train_dl_global[net_id], val_dl_global[net_id], test_dl, n_epoch,
                  args.lr, args.optimizer, device=device)
        # logger.info("net %d final val acc %f" % (net_id, testacc))
        # print("net : {} testacc: {}".format(net_id, testacc))
        # avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    # avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = nets
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=None,
                             device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global,
                                                             train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                                             device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch,
                                                        args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None,
                         round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch,
                                           args.lr,
                                           args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map


def Compute_alpha(args, dist_matrix, sample_ratio, lamda):
    """
    compute q vector and alpha matrix
    """
    q = np.array([args.n_parties] * args.n_parties)
    alpha = np.zeros((args.n_parties, args.n_parties))

    for i in range(args.n_parties):
        sort_idxs = np.argsort(dist_matrix[i])
        sort_sample_ratio = [sample_ratio[sort_idxs[i]] for i in range(args.n_parties)]
        sort_distance = np.sort(dist_matrix[i])

        for t in range(1, args.n_parties + 1):
            a = sum(sort_sample_ratio[:t])
            b = -2 * sum((sort_sample_ratio * sort_distance)[:t])
            c = sum((sort_sample_ratio * sort_distance * sort_distance)[:t]) - np.square(lamda)
            delta = b * b - 4 * a * c
            if delta < 0 or (-b + math.sqrt(delta)) / (2 * a) < sort_distance[t - 1]:
                q[i] = t - 1
                break

        a = sum(sort_sample_ratio[:q[i]])
        b = -2 * sum((sort_sample_ratio * sort_distance)[:q[i]])
        c = sum((sort_sample_ratio * sort_distance * sort_distance)[:q[i]]) - np.square(lamda)
        zeta = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        tmp = np.array([zeta - sort_distance[j] for j in range(args.n_parties)])
        for j in range(args.n_parties):
            alpha[i, sort_idxs[j]] = sort_sample_ratio[j] * (zeta - sort_distance[j]) / sum(
                (sort_sample_ratio * tmp)[:q[i]])

    alpha = np.maximum(alpha, 0)
    return q, alpha


def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # if client_id==0:
        #     print(f'Aggregation weight: {aggregation_weight_vector}. Summation: {aggregation_weight_vector.sum()}')

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
            cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))

    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

    return cluster_model_vectors


def weight_flatten(model):  # 只是比较全连接层
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[
                key]  # 算出各个客户端模型在本地训练开始时（全局模型下发）和结束时的差距
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    # print("model_similarity_matrix" ,model_similarity_matrix)
    return model_similarity_matrix


def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1,
                                 similarity_matric):
    # index_clientid = torch.tensor(list(map(int, list(nets_this_round.keys()))))     # for example, client 'index_clientid[0]'s model difference vector is model_difference_matrix[0]
    index_clientid = list(nets_this_round.keys())
    # model_difference_matrix = cal_model_difference(index_clientid, nets_this_round, nets_param_start, difference_measure)
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw,
                                                          similarity_matric)  # 得到了各个model这次训练完成后模型参数的之间的cosine
    # model_difference_matrix_numpy = model_difference_matrix.numpy()+1
    # print(model_difference_matrix_numpy)
    # save_heatmap(args, model_difference_matrix.numpy()+1)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lambda_1,
                                                    fed_avg_freqs)
    # print(f'Model difference: {model_difference_matrix[0]}')
    # print(f'Graph matrix: {graph_matrix}')
    return graph_matrix


def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl,
                          data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    global cluster_model
    for net_id, net in nets_this_round.items():

        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        # if net_id in benign_client_list:
        #
        #     val_acc = compute_accuracy(net, val_local_dls[net_id], device=device)
        #     personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,
        #                                                                               device=device)
        #
        #     if val_acc > best_val_acc_list[net_id]:
        #         best_val_acc_list[net_id] = val_acc
        #         best_test_acc_list[net_id] = personalized_test_acc
        #     print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(
        #         net_id, personalized_test_acc, generalized_test_acc))
        #
        #     logger.info('>> Pre-Training val accuracy: {}'.format(val_acc))
        #     logger.info('>> Pre-Training personalized_test accuracy: {}'.format(personalized_test_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                  weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id].to(device)

        net.to(device)
        net.train()

        if type(train_local_dl) == type([1]):
            pass
        else:
            train_local_dl = [train_local_dl]

        # writer = SummaryWriter()

        for epoch in range(args.epochs):
            epoch_loss_collector = []
            for tmp in train_local_dl:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    loss = criterion(out, target)

                    if round > 0:
                        flatten_model = []
                        for param in net.parameters():
                            flatten_model.append(param.reshape(-1))
                        flatten_model = torch.cat(flatten_model)
                        loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(
                            flatten_model)  # 余弦相似度
                        loss2.backward()

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

        # if net_id in benign_client_list:
        #     train_acc = compute_accuracy(net, train_local_dl, device=device)
        #     val_acc = compute_accuracy(net, val_local_dls[net_id], device=device)
        #     personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,
        #                                                                               device=device)
        #
        #     if val_acc > best_val_acc_list[net_id]:
        #         best_val_acc_list[net_id] = val_acc
        #         best_test_acc_list[net_id] = personalized_test_acc
        #     print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(
        #         net_id, personalized_test_acc, generalized_test_acc))
        #
        #     logger.info('>> clint %s, Training accuracy: %f' % (net_id, train_acc))
        #     logger.info('>> clint %s, Test accuracy: %f' % (net_id, val_acc))
        #     logger.info('>> clint %s, personalized_test accuracy: %f' % (net_id, personalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


def local_eval_pfedcollab(args, net, dl_local, loss_func, metric_func, domain_id):
    net.eval()
    net.to(device)
    total_examples, total_loss, total_metric = 0, 0, 0

    total_examples, total_loss, total_metric = 0, 0, 0

    with torch.no_grad():
        for *X, Y in dl_local:
            X = [x.to(args.device) for x in X]
            Y = Y.to(args.device)
            D = torch.ones_like(Y) * int(domain_id)  # domain id

            # get prediction
            logits = net(*X, Y)
            loss = loss_func(logits, D)
            metric = metric_func(logits, D)

            num_examples = len(X[0])
            total_examples += num_examples

            total_loss += loss.item() * num_examples

            total_metric += metric.item() * num_examples

    avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

    return avg_loss, avg_metric


def local_train_pfedcollab(args, net, train_dl_local, loss_func, metric_func, num_iters, domain_id):
    """
    The server send the {model}, {loss_func}, and {optimizer} to client.
    The client will train the {model} with given configureation for {num_iters} rounds.
    """
    net.to(device)
    net.train()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                               weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                               weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)

    iterator = iter(train_dl_local)
    total_examples, total_loss, total_metric = 0, 0, 0
    for it in range(num_iters):

        # Get a batch of data (may iterate the dataset for multiple rounds)
        try:
            *X, Y = next(iterator)
        except StopIteration:
            iterator = iter(train_dl_local)  # reset iterator, dataset is shuffled
            *X, Y = next(iterator)

        X = [x.to(args.device) for x in X]
        Y = Y.to(args.device)
        D = torch.ones_like(Y) * int(domain_id)  # domain id

        # get prediction
        logits = net(*X, Y)

        loss = loss_func(logits, D)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # record the loss and accuracy
            num_examples = len(X[0])
            total_examples += num_examples

            total_loss += loss.item() * num_examples

            metric = metric_func(logits, D)
            total_metric += metric.item() * num_examples

    avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

    return avg_loss, avg_metric

    # if type(train_dl_local) == type([1]):
    #     pass
    # else:
    #     train_dataloader = [train_dl_local]
    # total_examples, total_loss, total_metric = 0, 0, 0
    #
    # for tmp in train_dataloader:
    #     for batch_idx, (x, target) in enumerate(tmp):
    #         x, target = x.to(device), target.to(device)
    #
    #         optimizer.zero_grad()
    #         x.requires_grad = True
    #         target.requires_grad = False
    #         D = torch.ones_like(target) * int(domain_id)
    #         target = target.long()
    #
    #         out = net(x)
    #         linear2 = nn.Linear(10, 1).cuda()
    #         logits = linear2(out)
    #         loss = loss_func(logits, D)
    #
    #         loss.backward()
    #         optimizer.step()
    #         with torch.no_grad():
    #             # record the loss and accuracy
    #             num_examples = len(x)
    #             total_examples += num_examples
    #             total_loss += loss.item() * num_examples
    #
    #             metric = metric_func(logits, D)
    #             total_metric += metric.item() * num_examples
    #
    # avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
    #
    # return avg_loss, avg_metric


def state_to_tensor(model_state_dict):
    """
    Convert a state dict to a concatenated tensor
    Note: it is deep copy, since torch.cat is deep copy
    :param model_state_dict:
    :return:
    """
    tensor = [t.view(-1) for t in model_state_dict.values()]
    tensor = torch.cat(tensor)
    return tensor


def tensor_to_state(tensor, model_state_dict_template):
    """
    Convert a tensor back to state dict.
    Note: apply deepcopy inside the function. Only use the input state dict as a template
    :param model_state_dict:
    :return:
    """
    curr_idx = 0
    model_state_dict = copy.deepcopy(model_state_dict_template)
    for key, value in model_state_dict.items():
        numel = value.numel()
        shape = value.shape
        model_state_dict[key].copy_(tensor[curr_idx:curr_idx + numel].view(shape))
        curr_idx += numel

    return model_state_dict


def estimate_all(args, train_dl_global, net):
    shape_in = args.shape_in
    shape_out = args.shape_out
    num_label = max(2, shape_out)
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        model_class = MLPDivergenceEstimator
    else:
        model_class = MLPDivergenceEstimator

    model = model_class(feature_dim=shape_in, label_dim=num_label)

    metric_func = lambda logits, target: logits.ge(0).eq(target).float().mean()
    loss_func = lambda output, target: F.binary_cross_entropy_with_logits(output.view(-1), target.view(-1).float())

    model.to(args.device)

    # disc_type = args.divergence
    # 选择disc_type = 'C'
    num_rounds = 10

    # initialize an empty pairwise discrepancy matrix
    disc_matrix = torch.zeros((args.n_parties, args.n_parties))
    init_state = copy.deepcopy(model.state_dict())
    num_iters = 1
    for i, j in tqdm(combinations(range(args.n_parties), 2)):  # combinations(range(num_clients), 2)来生成所有可能的客户端对
        best_metric = - np.inf
        rounds_no_improve = 0
        client_i, client_j = i, j
        state_g = copy.deepcopy(init_state)
        # ######## ######## TRAINING ######## ########
        for r in range(num_rounds):
            # ======== ======== client i ======== ========
            model.load_state_dict(state_dict=state_g, strict=False)
            loss_i, metric_i = local_train_pfedcollab(args, model, train_dl_global[client_i], loss_func, metric_func,
                                                      num_iters, 0)
            state_i = copy.deepcopy(model.state_dict())
            # ======== ======== client j ======== ========
            model.load_state_dict(state_dict=state_g, strict=False)
            loss_j, metric_j = local_train_pfedcollab(args, model, train_dl_global[client_j], loss_func, metric_func,
                                                      num_iters, 1)
            state_j = copy.deepcopy(model.state_dict())
            # ======== ======== server ======== ========
            # 在服务器端比较两个client更新的模型参数，完全拉成一维向量进行比较
            # aggregate and get the new global model
            tensor_i, tensor_j = state_to_tensor(state_i), state_to_tensor(state_j)
            tensor_g = (tensor_i + tensor_j) / 2
            state_g = tensor_to_state(tensor_g, model_state_dict_template=global_model.state_dict())
            # print(state_g)
            # evaluation divergence
            if r % 10 == 0:
                model.load_state_dict(state_dict=state_g, strict=False)
                loss_i, metric_i = local_eval_pfedcollab(args, model, val_dl_global[client_i], loss_func, metric_func,
                                                         0)
                loss_j, metric_j = local_eval_pfedcollab(args, model, val_dl_global[client_j], loss_func, metric_func,
                                                         1)
                loss, metric = loss_i + loss_j, metric_i + metric_j
            else:
                loss_i, metric_i = local_eval_pfedcollab(args, model, train_dl_global[client_i], loss_func, metric_func,
                                                         0)
                loss_j, metric_j = local_eval_pfedcollab(args, model, train_dl_global[client_j], loss_func, metric_func,
                                                         1)
                loss, metric = loss_i + loss_j, metric_i + metric_j
            if metric > best_metric:
                best_metric = metric
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1
                if 100 <= rounds_no_improve:
                    break

        # ######## ######## TESTING ######## ########

        # model.load_state_dict(state_dict=state_g, strict=False)
        #
        # loss_i, metric_i = client_i.local_eval(model, loss_func, metric_func, 'valid', 0)
        # loss_j, metric_j = client_j.local_eval(model, loss_func, metric_func, 'valid', 1)
        #
        # loss = loss_i + loss_j
        # metric = metric_i + metric_j

        metric = best_metric
        # Algorithm 1 最后一行

        metric = metric - 1  # save 1/2 divergence

        tqdm.write("%d %d, metric:%f " % (i, j, metric))

        disc_matrix[i, j] = metric
        disc_matrix[j, i] = metric

    return disc_matrix


def choice_to_fixed_alpha(choice, beta):
    num_clients = len(choice)
    alpha = torch.zeros(num_clients, num_clients)
    alpha[choice.view(-1, 1) == choice.view(1, -1)] = 1.0
    alpha = alpha * beta
    alpha = alpha / alpha.sum(dim=1, keepdims=True)
    return alpha


def evaluate_with_fixed_weight(choice, disc, m, beta, C):
    alpha = choice_to_fixed_alpha(choice, beta)
    errors = evaluate(alpha, disc, m, beta, C)
    return errors


def evaluate(alpha, disc, m, beta, C):
    gen_errors = torch.sqrt((torch.square(alpha) / beta).sum(dim=1) / m)
    shift_errors = (alpha * disc).sum(dim=1)

    errors = C * gen_errors + shift_errors
    return errors


def reduction(errors, beta, typ='weighted_average'):
    if typ == 'unweighted_average':
        return errors.mean()
    elif typ == 'weighted_average':
        return (errors * beta).sum()
    elif typ == 'log_sum_exp':
        return torch.log(torch.exp(errors).sum())


def discrete_solver(disc, m, beta, C, init='local', max_iter=100, shuffle=True):
    log = []
    N = len(beta)  # num clients

    # initialize
    if init == 'local':
        choice = torch.arange(len(beta))
    else:
        choice = init
        # raise NotImplementedError('Unknown initialization. ')

    for it in tqdm(range(max_iter)):
        if shuffle:
            cids = torch.randperm(N)
        else:
            cids = torch.arange(N)

        earlystop = True

        for query_id in cids:
            # print('Now query ID:', query_id, choices)
            c_prev = choice[query_id].item()
            errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
            error_prev = reduction(errors, beta, typ='unweighted_average')
            log.append(error_prev.cpu().numpy())

            for c in range(N):
                choice[query_id] = c
                errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
                error = reduction(errors, beta, typ='unweighted_average')

                if error < error_prev:
                    c_prev = c
                    error_prev = error
                    earlystop = False

            choice[query_id] = c_prev

        if earlystop:
            break
    # print(choice)
    # choice = torch.LongTensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
    error = reduction(errors, beta, typ='unweighted_average')

    return choice, error, log


def choice_to_collab_list(choice):
    choice = choice.numpy()
    clusters = set(choice)
    collab = []
    for cluster_id in clusters:
        collab.append(list(np.where(choice == cluster_id)[0]))

    return collab


def get_race_distance(a, b):
    return np.linalg.norm(a - b, axis=1).mean()


class SRP_Gaussin_torch():
    def __init__(self, K, R, d, seed):
        self.N = K * R  # number of hashes
        self.d = d  # data dimension
        self.K = K
        self.R = R
        np.random.seed(seed)
        self.W = torch.from_numpy(np.random.normal(size=(self.N, d))).float()
        self.powersOfTwo = torch.from_numpy(np.array([2 ** i for i in range(self.K)])).float()

    def hash(self, x, device):
        output = torch.sign(torch.matmul(x.to(torch.float32).to(device), self.W.to(device).T))
        output = torch.gt(output, 0).float()
        output = output.reshape(-1, self.R, self.K)
        return torch.matmul(output, self.powersOfTwo.to(device)).int().cpu().numpy()


from numba import guvectorize
from numba import float64, intp


class RACE_SRP():
    @guvectorize([(intp[:, :], float64[:], float64[:, :], float64[:, :])], '(n,l1),(l2),(m,k)->(m,k)',
                 target="parallel", nopython=True, cache=True)
    def increasecount(hashcodes, alpha, zeros, out):
        out[:, :] = 0.0
        for i in range(out.shape[0]):
            for j in range(hashcodes.shape[0]):
                out[i, hashcodes[j, i]] += alpha[j]

    def __init__(self, repetitions, num_hashes, dimension, hashes, dtype=np.float32):
        self.dtype = dtype
        self.R = repetitions  # number of ACEs (rows) in the array
        self.W = 2 ** num_hashes  # range of each ACE (width of each row)
        self.K = num_hashes
        self.D = dimension
        self.N = 0
        self.counts = np.zeros((self.R, self.W), dtype=self.dtype)
        self.hashes = hashes

    # increase count(weight) for X (batchsize * dimension)
    def add(self, X, alpha, device):
        self.N += X.shape[0]
        hashcode = self.hashes.hash(X, device)
        if (device == "cuda"):
            hashcode = hashcode.cpu().numpy()
        self.counts = self.increasecount(hashcode, alpha, self.counts)
        pass

    def print_table(self):
        print(self.counts.round(decimals=2))


def sketch_input(K, R, d, hashes, localdata_idx, train_dl):
    race = RACE_SRP(R, K, d, hashes)
    # need data loader to apply same transform
    original_dataset = train_dl.dataset
    # original_dataset = original_dataset.data.numpy()
    alpha = np.ones(len(original_dataset))
    temp_loader = data.DataLoader(dataset=original_dataset, batch_size=len(original_dataset), shuffle=False)
    for x, _ in temp_loader:
        if x.dim() > 2:
            x = x.reshape(len(x), -1)
        race.add(x, alpha, device)
        break
    return race.counts, race.N


def save_heatmap(args, relevance_vectors):
    logger.info('>>Solved relevance:' + str(relevance_vectors))
    sys.exit("Terminating the program")


# def save_heatmap(args, relevance_vectors):
#     if isinstance(data, np.ndarray):
#         pass
#     else:
#         distance_matrix = np.array(relevance_vectors)
#     sns.heatmap(distance_matrix, annot=False, cmap='viridis', cbar=True, square=True)
#     # plt.title('Distance Matrix Heatmap')
#     plt.xlabel('Client Index')
#     plt.ylabel('Client Index')
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # plt.show()
#     if args.partition == 'homo':
#         if args.noise == 0.1:
#             path = './figures/heatmap/' + args.alg + '/noise.svg'
#         else:
#             path ='./figures/heatmap/' + args.alg + '/IID.svg'
#     else:
#         path ='./figures/heatmap/' + args.alg + '/' + args.partition + '.svg'
#     plt.savefig(os.path.join(current_dir, path))
#     sys.exit("Terminating the program")
def js_divergence(p, q):

    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    m = 0.5 * (p + q)
    log_m = torch.log(m)


    kl_p = F.kl_div(log_m, p, reduction='batchmean')
    kl_q = F.kl_div(log_m, q, reduction='batchmean')


    return 0.5 * (kl_p + kl_q)

if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'

    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.logdir, log_path), mode='w')],
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    for arg, value in vars(args).items():
        logging.info("参数 %s: %s", arg, value)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    data_distributions = np.zeros((args.n_parties, len(np.unique(y_train))))
    for row in traindata_cls_counts:
        for col in traindata_cls_counts[row]:
            data_distributions[row, col] = traindata_cls_counts[row][col]

    for row in range(len(data_distributions)):
        sum_row = sum(data_distributions[row])
        for col in range(len(data_distributions[row])):
            data_distributions[row][col] = data_distributions[row][col] / sum_row

    n_classes = len(np.unique(y_train))

    train_dl_global, val_dl_global, test_dl_global = get_dataloader(args, args.dataset,
                                                                    args.datadir,
                                                                    args.batch_size,
                                                                    32,
                                                                    dataidxss=net_dataidx_map)

    print("len train_dl_global:", len(train_dl_global))

    data_size = len(val_dl_global[0])

    print("-" * 80)
    print(args.alg)
    print(args.partition)
    print(args.dataset)

    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, data_distributions, train_dl_global, val_dl_global,
                            test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
    elif args.alg == 'pFedSV':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        total_params = 0
        total_size = 0
        for param in nets[0].parameters():
            num_params = param.numel()
            total_params += num_params
            total_size += num_params * param.element_size()
        print(f"模型参数总大小: {total_size / 1024 / 1024:.2f} MB")

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        local_models = []
        best_val_acc_list, best_test_acc_list = [], []
        relevance_vectors = []

        for i in range(args.n_parties):
            best_val_acc_list.append(0)
            best_test_acc_list.append(0)
            # clients’ relevence vector
            relevance_vectors.append([0 for p in range(args.n_parties)])

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            next_nets = local_train_pFedSV_net(nets, selected, args, train_dl_global, val_dl_global, relevance_vectors,
                                               test_dl=test_dl_global, device=device)
            # if round == 9:
            #     save_heatmap(args, relevance_vectors)

            nets = next_nets
            # local_train_net(nets, selected, args, train_dl_global, val_dl_global, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)

    elif args.alg == 'pFedJS':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]

        local_models = []
        best_val_acc_list, best_test_acc_list = [], []
        js_matrix = [[0 for _ in range(args.n_parties)] for _ in range(args.n_parties)]
        start_time = time.time()

        if args.use_feature:
            print("\n!!!\n use feature !!!\n!!!")
            for i in range(args.n_parties):
                for j in range(i+1, args.n_parties):
                    flattened_images1 = []
                    flattened_images2 = []
                    dataset1 = train_dl_global[i]
                    dataset2 = train_dl_global[j]

                    for data in dataset1:
                        images, _ = data
                        flattened_batch = images.flatten(start_dim=1)
                        flattened_images1.append(flattened_batch)

                    for data in dataset2:
                        images, _ = data
                        flattened_batch = images.flatten(start_dim=1)
                        flattened_images2.append(flattened_batch)

                    all_flattened_images1 = torch.cat(flattened_images1, dim=0)
                    all_flattened_images2 = torch.cat(flattened_images2, dim=0)

                    feature1 = all_flattened_images1.flatten()
                    feature2 = all_flattened_images2.flatten()

                    total1 = torch.sum(feature1)
                    total2 = torch.sum(feature2)

                    normalized_feature1 = feature1 / total1
                    normalized_feature2 = feature2 / total2

                    if len(normalized_feature1)!=len(normalized_feature2):
                        len1 = normalized_feature1.size(0)
                        len2 = normalized_feature2.size(0)
                        min_len = min(len(normalized_feature1), len(normalized_feature2))
                        if len1 > min_len:
                            normalized_feature1 = normalized_feature1[-min_len:]  # 从后面开始截断
                        if len2 > min_len:
                            normalized_feature2 = normalized_feature2[-min_len:]  # 从后面开始截断
                    JS_divergence = js_divergence(normalized_feature1, normalized_feature2)
                    js_matrix[j][i] = js_matrix[i][j] = min(math.sqrt(max(JS_divergence, 0) / 2),
                                          1 - 0.5 * math.exp(-max(JS_divergence, 0)))


        else:
            for i in range(args.n_parties):
                for j in range(args.n_parties):
                    dataset1 = train_dl_global[i]
                    dataset2 = train_dl_global[j]
                    count1 = Counter(dataset1.dataset.labels.tolist())
                    count2 = Counter(dataset2.dataset.labels.tolist())
                    label_ratio1 = torch.tensor(
                        [(count1[idx] + 1) / (dataset1.dataset.labels.shape[0] + args.n_parties) for idx in
                         range(args.n_parties)])  # smoothness
                    label_ratio2 = torch.tensor(
                        [(count2[idx] + 1) / (dataset2.dataset.labels.shape[0] + args.n_parties) for idx in
                         range(args.n_parties)])
                    JS_divergence = 0.5 * kl_div(label_ratio1.log(), (label_ratio1 + label_ratio2) / 2,
                                                 reduction='sum') + 0.5 * kl_div(label_ratio2.log(),
                                                                                 (label_ratio1 + label_ratio2) / 2,
                                                                                 reduction='sum')
                    js_matrix[i][j] = min(math.sqrt(abs(JS_divergence / 2)), 1 - 0.5 * math.exp(-JS_divergence))


        # save_heatmap(args, js_matrix)
        num_samples = np.array([len(train_dl_global[i]) for i in range(len(train_dl_global))])
        sample_ratio = np.divide(num_samples, sum(num_samples))
        Q, alpha = Compute_alpha(args, js_matrix, sample_ratio, lamda=args.lamda)
        print(alpha)
        for i in range(args.n_parties):
            best_val_acc_list.append(0)
            best_test_acc_list.append(0)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))

            ini_net = nets[0]
            sum_w = ini_net.state_dict()
            model_this_round_list = []
            for model_num in range(len(alpha)):
                for net_id, net in enumerate(nets.values()):
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            sum_w[key] = net_para[key] * float(alpha[model_num][net_id])
                    else:
                        for key in net_para:
                            sum_w[key] += net_para[key] * float(alpha[model_num][net_id])
                ini_net.load_state_dict(sum_w)
                model_this_round_list.append(ini_net)
            nets = {k: model_this_round_list[k] for k in range(len(model_this_round_list))}
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"程序运行时间: {elapsed_time} 秒")
            for net_idx in nets:
                nets[net_idx].to(device)
            next_nets = local_train_net(nets, selected, args, data_distributions, train_dl_global, val_dl_global,
                                        test_dl=test_dl_global, device=device)

            nets = next_nets
            # local_train_net(nets, selected, args, train_dl_global, val_dl_global, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)


    elif args.alg == 'pFedgraph':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_parameters = global_model.state_dict()

        if args.is_same_initial:  # 这里需要true
            for net_id, net in nets.items():
                net.load_state_dict(global_parameters)

        local_models = []
        best_val_acc_list, best_test_acc_list = [], []
        dw = []

        for i in range(args.n_parties):
            best_val_acc_list.append(0)
            best_test_acc_list.append(0)
            # clients’ relevence vector
            dw.append({key: torch.zeros_like(value) for key, value in nets[i].named_parameters()})

        graph_matrix = torch.ones(len(nets), len(nets)) / (len(nets) - 1)  # Collaboration Graph
        graph_matrix[range(len(nets)), range(len(nets))] = 0

        cluster_model_vectors = {}
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))
            arr = np.arange(args.n_parties)
            benign_client_list = arr.tolist()
            nets_param_start = {k: copy.deepcopy(nets[k]) for k in arr}

            mean_personalized_acc = local_train_pfedgraph(args, round, nets, cluster_model_vectors, train_dl_global,
                                                          val_dl_global, test_dl_global, data_distributions,
                                                          best_val_acc_list, best_test_acc_list, benign_client_list)
            start_time = time.time()
            total_data_points = sum([len(net_dataidx_map[k]) for k in benign_client_list])
            fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in benign_client_list}

            # manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)  # 输入为这轮开始时的model参数以及经过local_train_pfedgraph的本地参数

            graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets, global_parameters, dw, fed_avg_freqs,
                                                        args.alpha,
                                                        args.difference_measure)  # Graph Matrix is not normalized yet
            cluster_model_vectors = aggregation_by_graph(args, graph_matrix, nets,
                                                         global_parameters)  # Aggregation weight is normalized here
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"程序运行时间: {elapsed_time} 秒")
            if round == 5:
                pass
            if round == 10:
                pass
            if round == 20:
                pass
            if round ==30:
                pass
            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)

    elif args.alg == 'CE':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        relevance_vectors = []
        for i in range(args.n_parties):
            relevance_vectors.append([0 for p in range(args.n_parties)])
        users_used = [i for i in range(args.n_parties)]
        # net_dataidx_map, traindata_cls_counts, data_distributions, train_dl_global, val_dl_global, test_dl_global
        args.train_pt = True
        start_time = time.time()
        model = Training_all(args, logger, traindata_cls_counts, train_dl_global, val_dl_global, test_dl_global,
                             users_used=users_used)
        model.train_pt(0)
        Benefit_Matrix = model.benefit(users_used)

        epsilon = 1 / args.n_parties - (0.5 / args.n_parties)
        Benefit_Matrix = Benefit_Matrix * (Benefit_Matrix > epsilon)

        print('Benefit_Matrix')
        print(Benefit_Matrix)

        logger.info('Benefit_Matrix')
        logger.info(Benefit_Matrix)

        for i in range(len(Benefit_Matrix)):
            for j in range(len(Benefit_Matrix[i])):
                if Benefit_Matrix[i][j] != 0:
                    if args.SCFL_data_num:
                        relevance_vectors[i][j] = len(net_dataidx_map[j])
                    else:
                        relevance_vectors[i][j] = 1

        for idx in range(len(relevance_vectors)):
            relevance_vectors[idx] = [x / sum(relevance_vectors[idx]) for x in relevance_vectors[idx]]

        # save_heatmap(args, relevance_vectors)

        logger.info('relevance_vectors')
        logger.info(relevance_vectors)
        print(relevance_vectors)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行时间: {elapsed_time} 秒")

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            next_nets = local_train_FedCollab_net(nets, selected, args, train_dl_global, val_dl_global,
                                                  relevance_vectors,
                                                  test_dl=test_dl_global, device=device)

            nets = next_nets
            # local_train_net(nets, selected, args, train_dl_global, val_dl_global, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)


    elif args.alg == 'pFedCollab':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_parameters = global_model.state_dict()

        train_dl_global_dict = {i: train_dl_global[i] for i in range(len(train_dl_global))}

        if args.is_same_initial:  # 这里需要true
            for net_id, net in nets.items():
                net.load_state_dict(global_parameters)

        local_models = []
        best_val_acc_list, best_test_acc_list = [], []
        relevance_vectors = []

        for i in range(args.n_parties):
            best_val_acc_list.append(0)
            best_test_acc_list.append(0)
            # clients’ relevence vector
            relevance_vectors.append([0 for p in range(args.n_parties)])
        start_time = time.time()
        ms = torch.Tensor([len(data.dataset) for data in train_dl_global])
        m = ms.sum()
        beta = ms / m
        disc_matrix = estimate_all(args, train_dl_global, global_models)

        # save_heatmap(args, disc_matrix)
        # disc_matrix = torch.randn(10, 10)
        # 开始第三部分 collab_opt
        disc = torch.max(torch.Tensor([0]), disc_matrix)
        # if True:
        #     plt.imshow(disc)
        #     plt.show()

        solver = discrete_solver
        low = np.inf
        best = None
        logs = []
        # import time
        # aa = time.time()
        for i in range(100):
            choice, error, log = solver(disc, m, beta, C=1)
            if error < low:
                low = error
                best = choice

            logs.append(log)

        choice = best

        collab = choice_to_collab_list(choice)

        print('Solved collaboration:', collab)
        logger.info('>>Solved collaboration:' + str(collab))

        print('Loss:', error * len(beta))
        logger.info("in comm round:" + str(round))

        for idx, coalition in enumerate(collab):
            print('Running Coalition %d / %d' % (idx + 1, len(collab)))
            logger.info('Running Coalition %d / %d' % (idx + 1, len(collab)))
            print(coalition)
            logger.info('coalition %d ' + str(coalition))

            # sub_datasets = {cid: dataset for (cid, dataset) in train_dl_global_dict.items() if cid in coalition}

            for idx in range(len(relevance_vectors)):
                if idx in coalition:
                    for index in coalition:
                        if args.SCFL_data_num:
                            relevance_vectors[idx][index] = len(net_dataidx_map[index])
                        else:
                            relevance_vectors[idx][index] = 1

        for idx in range(len(relevance_vectors)):
            relevance_vectors[idx] = [x / sum(relevance_vectors[idx]) for x in relevance_vectors[idx]]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行时间: {elapsed_time} 秒")

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            next_nets = local_train_FedCollab_net(nets, selected, args, train_dl_global, val_dl_global,
                                                  relevance_vectors,
                                                  test_dl=test_dl_global, device=device)

            nets = next_nets
            # local_train_net(nets, selected, args, train_dl_global, val_dl_global, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)


    elif args.alg == 'RACE':
        # net_dataidx_map, traindata_cls_counts, data_distributions, train_dl_global, val_dl_global, test_dl_global
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        arr = np.arange(args.n_parties)
        raceK, raceR, IN = get_KRI(args)
        hashes = SRP_Gaussin_torch(raceK, raceR, IN, seed)
        sketch_buffer = []
        sketch_global_N = 0
        start_time = time.time()
        for client in arr:
            device_s, device_n = sketch_input(raceK, raceR, IN, hashes, net_dataidx_map[client],
                                              train_dl_global[client])
            memory_size_bytes = device_s.nbytes
            memory_size_mb = memory_size_bytes / (1024 * 1024)
            # print(f"Memory Size: {memory_size_mb:.6f} MB")

            sketch_buffer.append(np.expand_dims(device_s, axis=0))
            sketch_global_N += device_n
        global_sketch = np.concatenate(sketch_buffer, axis=0).sum(axis=0)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行时间: {elapsed_time} 秒")
        local_models = []
        best_val_acc_list, best_test_acc_list = [], []
        relevance_vectors = []
        distance_matrix = []

        for i in range(args.n_parties):
            best_val_acc_list.append(0)
            best_test_acc_list.append(0)
            # clients’ relevence vector
            relevance_vectors.append([0 for p in range(args.n_parties)])
            distance_matrix.append([0 for p in range(args.n_parties)])

        best_testacc = 0
        wait_round = 0
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print("in comm round:{}".format(round))
            for i in range(len(relevance_vectors)):
                for j in range(len(relevance_vectors[i])):
                    relevance_vectors[i][j] = 0
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            devices_random = np.random.choice(arr, int(3 * args.K), replace=False)
            # select device based on data distacne
            device_s = {}  # each device's sketch
            device_n = {}  # each device's # of data
            distance = {}
            score = {}

            # for i in range(args.n_parties):
            #     for j in range(args.n_parties):
            #         device_s[i], device_n[i] = sketch_input(raceK, raceR, IN, hashes, net_dataidx_map[i],
            #                                                         train_dl_global[i])
            #         device_s[j], device_n[j] = sketch_input(raceK, raceR, IN, hashes, net_dataidx_map[j],
            #                                                         train_dl_global[j])
            #         distance_matrix[i][j] = get_race_distance(device_s[i] / device_n[i], device_s[j] / device_n[j])
            # save_heatmap(args, distance_matrix)

            for dev_i in devices_random:
                device_s[dev_i], device_n[dev_i] = sketch_input(raceK, raceR, IN, hashes, net_dataidx_map[dev_i],
                                                                train_dl_global[dev_i])
                distance[dev_i] = get_race_distance(global_sketch / sketch_global_N, device_s[dev_i] / device_n[dev_i])

            np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
            for k, v in distance.items():
                score[k] = 1 / v + 1e-6
            prob = torch.Tensor(list(score.values())) / 1.0 + +1e-6
            prob = torch.nn.functional.softmax(prob, dim=0).data.numpy()
            prob /= prob.sum()
            if prob.sum != 1.0:
                prob[-1] = 1.0 - prob[:-1].sum()
            try:
                valid_id = np.random.choice(
                    list(score.keys()),
                    args.K,
                    p=prob,
                    replace=False,
                )
            except ValueError as e:
                try:
                    valid_id = np.random.choice(
                        list(score.keys()),
                        args.K - 1,
                        p=prob,
                        replace=False,
                    )
                except ValueError as e:
                    valid_id = np.random.choice(
                        list(score.keys()),
                        args.K - 2,
                        p=prob,
                        replace=False,
                    )

            for idx in range(len(relevance_vectors)):
                if idx in valid_id:
                    for index in valid_id:
                        if args.SCFL_data_num:
                            relevance_vectors[idx][index] = len(net_dataidx_map[index])
                        else:
                            relevance_vectors[idx][index] = 1
                if idx not in valid_id:
                    relevance_vectors[idx][idx] = 1

            for idx in range(len(relevance_vectors)):
                relevance_vectors[idx] = [x / sum(relevance_vectors[idx]) for x in relevance_vectors[idx]]

            next_nets = local_train_FedCollab_net(nets, selected, args, train_dl_global, val_dl_global,
                                                  relevance_vectors,
                                                  test_dl=test_dl_global, device=device)

            nets = next_nets
            for idx2 in devices_random:
                nets[idx2] = nets[valid_id[0]]

            if (round + 1) % args.round_calculate == 0:
                # train_acc_sum = 0
                # val_acc_sum = 0
                test_acc_sum = 0
                ptest_acc_sum = 0
                for idx in range(args.n_parties):
                    correct, total = 0, 0
                    # for net_id, net in nets_this_round.items():
                    val_local_dl = val_dl_global[idx]
                    train_local_dl = train_dl_global[idx]
                    net = nets[idx]
                    # train_acc = compute_accuracy(net, train_local_dl, device=device)
                    # val_acc = compute_accuracy(net, val_local_dl, device=device)
                    ptest_acc, test_acc = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx],
                                                                      device=device)
                    test_acc_sum = test_acc_sum + test_acc
                    ptest_acc_sum = ptest_acc_sum + ptest_acc
                    # train_acc_sum = train_acc_sum + train_acc
                    # val_acc_sum = val_acc_sum + val_acc
                # end_train_acc = train_acc_sum / args.n_parties
                # end_val_acc = val_acc_sum / args.n_parties
                end_test_acc = test_acc_sum / args.n_parties
                end_ptest_acc = ptest_acc_sum / args.n_parties

                # logger.info('>> Global Model Train accuracy: %f' % end_train_acc)
                # logger.info('>> Global Model Val accuracy: %f' % end_val_acc)
                logger.info('>> Global Model ptest accuracy: %f' % end_ptest_acc)
                logger.info('>> Global Model test accuracy: %f' % end_test_acc)
                # print("round {},train acc = {:.5f}".format(round, end_train_acc))
                # print("round {},val acc = {:.5f}".format(round, end_val_acc))
                print("round {},ptest acc = {:.5f}".format(round, end_ptest_acc))
                print("round {},test acc = {:.5f}".format(round, end_test_acc))
            print('-' * 80)

            # train_acc_sum = 0
            # val_acc_sum = 0
            # test_acc_sum = 0
            # for idx in valid_id:
            #     correct, total = 0, 0
            #     # for net_id, net in nets_this_round.items():
            #     val_local_dl = val_dl_global[idx]
            #     train_local_dl = train_dl_global[idx]
            #     net = nets[idx]
            #     train_acc = compute_accuracy(net, train_local_dl, device=device)
            #     val_acc = compute_accuracy(net, val_local_dl, device=device)
            #     test_acc, _ = compute_local_test_accuracy(net, test_dl_global, data_distributions[idx], device=device)
            #     test_acc_sum = test_acc_sum + test_acc
            #     train_acc_sum = train_acc_sum + train_acc
            #     val_acc_sum = val_acc_sum + val_acc
            # end_train_acc = train_acc_sum / len(valid_id)
            # end_val_acc = val_acc_sum / len(valid_id)
            # end_test_acc = test_acc_sum / len(valid_id)
            #
            # logger.info('devices_random K >> Global Model Train accuracy: %f' % end_train_acc)
            # logger.info('devices_random K >> Global Model Val accuracy: %f' % end_val_acc)
            # logger.info('devices_random K >> Global Model test accuracy: %f' % end_test_acc)
            # print("devices_random K round {},train acc = {:.5f}".format(round, end_train_acc))
            # print("devices_random K round {},val acc = {:.5f}".format(round, end_val_acc))
            # print("devices_random K round {},test acc = {:.5f}".format(round, end_test_acc))


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global,
                                    device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                     test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=test_dl_global, device=device)
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    # if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    # else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                 global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     moon_model=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        ptestacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer,
                                      device=device)

        logger.info("All in test acc: %f" % testacc)
