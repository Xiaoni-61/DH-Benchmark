for data in mnist cifar10
do
  for size in 16 32 64 128 256
  do
    for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
    do
      python ../experiments.py --model=simple-cnn \
        --dataset=$data \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=$size \
        --epochs=10 \
        --n_parties=10 \
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
    done
  done
done
	
