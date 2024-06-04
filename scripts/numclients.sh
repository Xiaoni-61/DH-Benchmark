for data in mnist cifar10
do
  for n in 10 20 30 40
  do
    for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
    do
      python ../experiments.py --model=simple-cnn \
        --dataset=$data \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=$n \
        --rho=0.9 \
        --mu=0.01 \
        --comm_round=50 \
        --partition=noniid-labeldir \
        --beta=0.5\
        --device='cuda:0'\
        --datadir='../data/' \
        --logdir='../logs/' \
        --noise=0\
        --sample=0\
        --init_seed=0

      python ../experiments.py --model=simple-cnn \
        --dataset=$data \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=$n \
        --rho=0.9 \
        --mu=0.01 \
        --comm_round=50 \
        --partition=homo \
        --beta=0.5\
        --device='cuda:0'\
        --datadir='../data/' \
        --logdir='../logs/' \
        --noise=0.1\
        --sample=0\
        --init_seed=0
    done
  done
done
	
