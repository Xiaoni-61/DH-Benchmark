# DH-Benchmark



This is the code of **paper Benchmarking Data Heterogeneity Evaluation Approaches for Personalized Federated Learning**

This code runs a benchmark for personalized federated learning algorithms to assess Data Heterogeneity Evaluation
Approaches. Specifically, We compared six algorithms (pFedSV/pFedJS/pFedgraph/FedCollab/RACE/CE), 3 types of non-IID settings (label
distribution skew, feature distribution skew & quantity skew) and 8 datasets (MNIST, Cifar-10, Fashion-MNIST, SVHN,
 FEMNIST, adult, rcv1, covtype).

We acknowledge the use of the NIID-Bench code framework in our design and express our gratitude for it. Code from: [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)

## Non-IID Settings

### Label Distribution Skew

* **Quantity-based label imbalance**: each party owns data samples of a fixed number of labels.
* **Distribution-based label imbalance**: each party is allocated a proportion of the samples of each label according to
  Dirichlet distribution.

### Feature Distribution Skew

* **Noise-based feature imbalance**: We first divide the whole dataset into multiple parties randomly and equally. For
  each party, we add different levels of Gaussian noises.
* **Real-world feature imbalance**: For FEMNIST, we divide and assign the writers (and their characters) into each party
  randomly and equally.

### Quantity Skew

* While the data distribution may still be consistent amongthe parties, the size of local dataset varies according to
  Dirichlet distribution.

## Usage

Here is one example to run this code:

```
python experiments.py --model=simple-cnn \
    --dataset=cifar10 \
    --alg=fedprox \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=10 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0 \
    --SCFL_data_num \
    --use_feature
```

| Parameter       | Description                                                                                                                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`         | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`.                                                                                                       |
| `dataset`       | Dataset to use. Options: `mnist`, `cifar10`, `fmnist`, `svhn`, `femnist`, `a9a`, `rcv1`, `covtype`. Default = `mnist`.                                                                        |
| `alg`           | The training algorithm. Options: `fedavg`, `fedprox`, `scaffold`, `fednova`, `moon`. Default = `fedavg`.                                                                                      |
| `lr`            | Learning rate for the local models, default = `0.01`.                                                                                                                                         |
| `batch-size`    | Batch size, default = `64`.                                                                                                                                                                   |
| `epochs`        | Number of local training epochs, default = `5`.                                                                                                                                               |
| `n_parties`     | Number of parties, default = `2`.                                                                                                                                                             |
| `rho`           | The parameter controlling the momentum SGD, default = `0`.                                                                                                                                    |
| `comm_round`    | Number of communication rounds to use, default = `50`.                                                                                                                                        |
| `partition`     | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `real`, `iid-diff-quantity`. Default = `homo` |
| `beta`          | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`.                                                                                       |
| `device`        | Specify the device to run the program, default = `cuda:0`.                                                                                                                                    |
| `datadir`       | The path of the dataset, default = `./data/`.                                                                                                                                                 |
| `logdir`        | The path to store the logs, default = `./logs/`.                                                                                                                                              |
| `noise`         | Maximum variance of Gaussian noise we add to local party, default = `0`.                                                                                                                      |
| `sample`        | Ratio of parties that participate in each communication round, default = `1`.                                                                                                                 |
| `init_seed`     | The initial seed, default = `0`.                                                                                                                                                              |
| `SCFL_data_num` | Consider the amount of data when weighting aggregation models.                                                                                                                                |
| `use_feature`   | Using feature instead of label to use pFedJS                                                                                                                                                  |                                                                                                                                                                                |


If you have any questions, you can contact lzl@bupt.edu.cn
