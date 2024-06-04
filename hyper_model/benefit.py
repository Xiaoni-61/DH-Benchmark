import argparse
import collections
import copy
import json
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from tqdm import trange
import pdb
from torch.utils.data import DataLoader, Dataset
from hyper_model.models import Hypernet, HyperMLP,HyperSimpleNet, SimpleNet, Basenet_cifar
from hyper_model.solvers import EPOSolver, LinearScalarizationSolver
from torch.autograd import Variable
import os
from sklearn.metrics import roc_auc_score
import time
import torch.utils.data as data

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from collections import OrderedDict

"""
training a personalized model for each client;
"""


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Training_all(object):
    def __init__(self, args, logger, traindata_cls_counts, train_dl_global, val_dl_global, test_dl_global,
                 users_used=None):
        self.device = args.device
        self.args = args
        self.test_train = False
        self.total_epoch = 10
        self.epochs_per_valid = 10
        self.target_usr = 2
        if users_used == None:
            self.users_used = [i for i in range(self.args.n_parties)]
        else:
            self.users_used = users_used
        self.all_users = [i for i in range(self.args.n_parties)]

        ################# DATA  #######################
        # self.dataset_test = Data.dataset_test
        # self.data_test = Data.dataset_test
        # self.dataset_train = Data.dataset_train

        self.dict_user_train = traindata_cls_counts
        # self.dict_user_test = Data.dict_user_test
        # self.dict_user_valid = Data.dict_user_valid

        self.train_loaders = train_dl_global
        self.valid_loaders = val_dl_global

        ################# model  #######################
        # if args.dataset == "synthetic1" or args.dataset == "synthetic2" :
        #     if args.baseline_type == 'ours':
        #         self.hnet = HyperSimpleNet( args, self.device, self.users_used)
        #     else:
        #         self.hnet = SimpleNet(args, self.device)
        # elif args.dataset == "eicu":
        #     if args.baseline_type == 'ours':
        #         self.hnet = Hypereicu(args=args, usr_used=self.users_used, device=self.device)
        #     else:
        #         self.hnet = Basenet(args)
        if args.dataset in ('mnist', 'femnist', 'fmnist', 'cifar10','svhn'):
            self.hnet = Hypernet(args=args, n_usrs=args.n_parties, device=self.device, n_classes=10,
                                 usr_used=self.users_used, n_hidden=2, spec_norm=False)
        elif args.dataset in ('a9a', 'rcv1', 'covtype'):
            if args.dataset =='a9a':
                in_dim =123
            elif args.dataset == 'covtype':
                in_dim = 54
            elif args.dataset == 'rcv1':
                in_dim = 47236
            self.hnet = HyperMLP(args=args, n_usrs=args.n_parties, usr_used=self.users_used, n_classes=10,device=self.device,
                                     input_dim=in_dim, hidden_dims=[100, 50], output_dim=2)




        self.hnet.to(self.device)

        self.optim = torch.optim.Adam(self.hnet.parameters(), lr=0.001)

        # if args.solver_type == "epo":
        # self.solver = EPOSolver(len(self.users_used))
        # elif args.solver_type == "linear":
        self.solver = LinearScalarizationSolver(len(self.users_used))
        self.logger = logger
        self.global_epoch = 0

    def all_args_save(self, args):
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    def train_input(self, usr_id):
        try:
            _, (X, Y) = self.train_loaders[usr_id].__next__()
        except StopIteration:
            if self.args.local_bs == -1:
                t1 = time.time()
                self.train_loaders[usr_id] = enumerate(
                    FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]),
                                   batch_size=len(self.dict_user_train[usr_id]), shuffle=True,
                                   num_workers=self.args.num_workers))
                t2 = time.time()
            else:
                self.train_loaders[usr_id] = enumerate(
                    FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]),
                                   batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers))
            _, (X, Y) = self.train_loaders[usr_id].__next__()
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y

    def valid_input(self, usr_id):
        try:
            _, (X, Y) = self.valid_loaders[usr_id].__next__()
        except StopIteration:
            self.valid_loaders[usr_id] = enumerate(
                FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_valid[usr_id]),
                               batch_size=min(512, len(self.dict_user_valid[usr_id])), shuffle=True,
                               num_workers=self.args.num_workers))
            _, (X, Y) = self.valid_loaders[usr_id].__next__()
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y

    def ray2users(self, ray):
        tmp_users_used = []
        tmp_ray = []
        for user_id, r in enumerate(ray):
            if r / ray[self.target_usr] >= 0.7:
                tmp_users_used.append(user_id)
                tmp_ray.append(r)
        return tmp_users_used, tmp_ray

    def acc_auc(self, prob, Y, is_training=True):
        if self.args.dataset == "adult" or self.args.dataset == "eicu":
            y_pred = prob.data >= 0.5
        elif self.args.dataset == "synthetic1" or self.args.dataset == "synthetic2":
            if is_training:
                return 0
            else:
                return 0, 0
        else:
            y_pred = prob.data.max(1)[1]
        users_acc = torch.mean((y_pred == Y).float()).item()

        if self.args.dataset == "eicu" and is_training:
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return users_auc

        elif is_training and self.args.dataset != "eicu":
            return users_acc

        elif self.args.dataset == "eicu" and not is_training:
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return users_acc, users_auc
        else:
            return users_acc, 0

    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl

    def train_pt(self, target_usr):  # training the Pareto Front using training data
        start_epoch = 0
        for iteration in range(start_epoch, 1):
            self.hnet.train()

            total_params = 0
            total_size = 0
            for param in self.hnet.parameters():
                num_params = param.numel()
                total_params += num_params
                total_size += num_params * param.element_size()
            print(f"hnet模型参数总大小: {total_size / 1024 / 1024:.2f} MB")


            losses = []
            accs = {}
            loss_items = {}

            ray = torch.from_numpy(
                np.random.dirichlet([0.1 for i in self.users_used], 1).astype(np.float32).flatten()).to(
                self.device)
            ray = ray.view(1, -1)

            for usr_id in self.users_used:

                dataset = self.train_loaders[usr_id].dataset
                dataset_size = len(dataset)
                new_dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
                for batch_idx, (x, target) in enumerate(new_dataloader):
                    x, target = x.to(self.args.device), target.to(self.args.device)
                    target = target.long()
                    pred, loss = self.hnet(x, target, usr_id, ray)
                    acc = self.acc_auc(pred, target)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            loss, alphas = self.solver(losses, ray,
                                       [p for n, p in self.hnet.named_parameters() if "local" not in n])
            # print("round:{} Loss!!: {}".format(iteration, loss))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    pass

    # if iteration % 1 == 0:
    #     _, _, loss_dict = self.valid()
    #
    #     self.logger.info('hyper iteration :{} losses: {} '.format(iteration, losses.data))

    def train_ray(self, target_usr):  ## searching the optimal model on the PF using the validation data

        start_epoch = 0
        for iteration in range(start_epoch, 5):
            self.hnet.train()

            dataset = self.valid_loaders[target_usr].dataset
            dataset_size = len(dataset)
            new_dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
            for batch_idx, (X, Y) in enumerate(new_dataloader):
                X = X.to(self.args.device)
                Y = Y.to(self.args.device)
                pred, loss = self.hnet(X, Y, target_usr)
                acc = self.acc_auc(pred, Y)
                self.optim.zero_grad()
                self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
                loss.backward()

            # self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * 0.01)
            self.hnet.input_ray.data = self.hnet.input_ray.data + (-self.hnet.input_ray.grad * 0.01)

            self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, 0.01, 1)
            self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]


        return self.hnet.input_ray.data

    def train_baseline(self):  ## training baselines (FedAve and Local)
        if self.args.baseline_type == "fedave":
            start_epoch = self.global_epoch
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(start_epoch, self.total_epoch):
                self.hnet.train()

                losses = []
                accs = {}
                loss_items = {}

                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id)
                    pred, loss = self.hnet(X, Y, usr_id)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

                loss = torch.mean(torch.stack(losses))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('fedaveloss', loss, iteration)
                if iteration % 1 == 0:
                    accs, aucs, _ = self.valid()
                    mean_acc = sum(accs.values()) / len(self.users_used)
                    self.logger.info('iteration: {}'.format(iteration))

                self.global_epoch += 1
                if (self.global_epoch + 1) % self.epochs_per_valid == 0:
                    self.valid()

            if self.args.tensorboard:
                writer.close()

        elif self.args.baseline_type == "local":
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(0, self.total_epoch):
                self.hnet.train()
                loss_items = {}

                X, Y = self.train_input(self.args.target_usr)

                pred, loss = self.hnet(X, Y, self.args.target_usr)
                acc = self.acc_auc(pred, Y)
                loss_items[str(self.args.target_usr)] = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('usr{}loss'.format(self.args.target_usr), loss, iteration)

                self.global_epoch += 1
                if iteration % 1 == 0:
                    if self.args.personal_init_epoch != 0:
                        self.logger.info('init iteration:{} '.format(iteration))


                    else:
                        self.logger.info('local iteration:{} '.format(iteration))

                if (self.global_epoch + 1) % self.epochs_per_valid == 0:
                    self.valid()
            if self.args.tensorboard:
                writer.close()

        else:
            self.logger.info("error baseline type")
            exit()

    def train(self):

        if self.args.train_baseline:
            self.train_baseline()

        else:
            if self.args.sample_ray == True:
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)

            else:
                results = {}
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)
                for usr in self.users_used:
                    accs, aucs, _ = self.valid(ray=None, target_usr=usr)
                    self.logger.info('usr :{},  weight: {}'.format(usr, self.hnet.input_ray.data[usr]))

                    if self.args.dataset == 'eicu':
                        results[usr] = aucs[str(usr)]
                    else:
                        results[usr] = accs[str(usr)]
                results = np.array(list(results.values()))

    def train_personal_hyper(self, user, users_used):
        self.args.train_baseline = True
        self.args.baseline_type = "local"
        self.total_epoch = self.args.personal_init_epoch
        self.train_baseline()
        self.hnet.init_ray(users_used.index(user))

        self.args.baseline_type = 'ours'

        for iteration in range(self.args.personal_epoch):

            # ----------------------train hyper---------------------------
            self.hnet.train()
            losses = []
            accs = {}
            loss_items = {}

            for usr_id in self.users_used:
                X, Y = self.train_input(usr_id)
                pred, loss = self.hnet(X, Y, usr_id)
                acc = self.acc_auc(pred, Y)
                accs[str(usr_id)] = acc
                losses.append(loss)
                loss_items[str(usr_id)] = loss.item()

            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.info("hyper iteration: {}, input_ray: {}.".format(iteration, input_ray_numpy))

            # ------------------------------train ray-----------------------------
            self.hnet.train()
            X, Y = self.valid_input(user)
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred, loss = self.hnet(X, Y, user)
            acc = self.acc_auc(pred, Y)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            loss.backward()

            self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
            self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1)
            self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

            self.logger.info('ray iteration: {},  ray: {}'.format(iteration, self.hnet.input_ray.data))

            if (iteration % 10 == 0):
                acc, auc, loss = self.personal_test(user)

    def benefit(self, usr_used):
        benefit_matrix = []
        i = 0
        print('users')
        print(usr_used)
        for usr in usr_used:
            self.hnet.init_ray(usr)
            self.args.target_usr = usr
            ray = self.train_ray(usr)
            # accs, aucs, _ = self.valid(ray=ray, target_usr=usr)
            # self.logger.info('usr: {}, acc: {}， aucs: {}, weight: {}'.format(usr, accs[str(usr)], aucs[str(usr)], ray))
            # self.logger.info(ray.cpu().numpy()[0])
            benefit_matrix.append(ray.cpu().numpy()[0])
            i += 1

        benefit_matrix = np.vstack(benefit_matrix)

        return benefit_matrix

    def test_input(self, data_loader):
        _, (X, Y) = data_loader.__next__()
        return X.to(self.device), Y.to(self.device)

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl

    def valid_baseline(self, model="baseline", target_usr=0, load=False, ckptname="last", train_data=False):
        with torch.no_grad():
            if train_data:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[idxs]),
                                                         batch_size=len(self.dict_user_train[idxs]), shuffle=False,
                                                         drop_last=False))
                                for idxs in range(self.args.num_users)]
            else:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                         batch_size=len(self.dict_user_test[idxs]), shuffle=False,
                                                         drop_last=False))
                                for idxs in range(self.args.num_users)]
            if load:
                self.load_hnet(ckptname)

            accs = {}
            loss_dict = {}
            loss_list = []
            aucs = {}

            for usr_id in self.users_used:
                X, Y = self.test_input(data_loaders[usr_id])
                pred, loss = self.hnet(X, Y, usr_id)
                acc, auc = self.acc_auc(pred, Y, is_training=False)
                accs[str(usr_id)] = acc
                loss_dict[str(usr_id)] = loss.item()
                loss_list.append(loss.item())
                aucs[str(usr_id)] = auc

            return accs, aucs, loss_dict

    def valid(self, model="hnet", ray=None, target_usr=0, load=False, ckptname="last", train_data=False):
        target_usr = self.target_usr
        self.hnet.eval()
        if self.args.train_baseline:
            accs, aucs, loss_dict = self.valid_baseline()
            return accs, aucs, loss_dict
        else:
            with torch.no_grad():
                if train_data:
                    data_loaders = [enumerate(
                        FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[idxs]),
                                       batch_size=len(self.dict_user_train[idxs]), shuffle=False, drop_last=False))
                        for idxs in range(self.args.num_users)]
                else:
                    data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                             batch_size=len(self.dict_user_test[idxs]), shuffle=False,
                                                             drop_last=False))
                                    for idxs in range(self.args.num_users)]
                if load:
                    self.load_hnet(ckptname)

                accs = {}
                loss_dict = {}
                loss_list = []
                aucs = {}

                for usr_id in self.users_used:
                    if self.test_train:
                        X, Y = self.train_input(usr_id)
                    else:
                        X, Y = self.test_input(data_loaders[usr_id])
                    pred, loss = self.hnet(X, Y, usr_id, ray)
                    acc, auc = self.acc_auc(pred, Y, is_training=False)
                    accs[str(usr_id)] = acc
                    loss_dict[str(usr_id)] = loss.item()
                    loss_list.append(loss.item())
                    aucs[str(usr_id)] = auc

                input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

                return accs, aucs, loss_dict

    def personal_test(self, user):
        with torch.no_grad():
            data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                     batch_size=len(self.dict_user_test[idxs]), shuffle=False,
                                                     drop_last=False))
                            for idxs in range(self.args.num_users)]
            X, Y = self.test_input(data_loaders[user])
            pred, loss = self.hnet(X, Y, user)
            acc, auc = self.acc_auc(pred, Y, is_training=False)
        return acc, auc, loss.item()

    def save_hnet(self, ckptname=None):
        states = {'epoch': self.global_epoch,
                  'model': self.hnet.state_dict(),
                  'optim': self.optim.state_dict(),
                  'input_ray': self.hnet.input_ray.data}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        os.makedirs(self.args.hnet_model_dir, exist_ok=True)
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        print(filepath)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> hnet saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))

    def load_hnet(self, ckptname='last'):
        if ckptname == 'last':
            ckpts = os.listdir(self.args.hnet_model_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.global_epoch = checkpoint['epoch']
            self.hnet.load_state_dict(checkpoint['model'])
            # self.hnet.input_ray.data = checkpoint["input_ray"].data.view(1, -1)
            self.optim.load_state_dict(checkpoint['optim'])
            self.logger.info("=> hnet loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))
