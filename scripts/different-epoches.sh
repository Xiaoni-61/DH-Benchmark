for data in mnist cifar10 svhn fmnist
do
  for e in 20 40 50
  do
    for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
    do
      for par in noniid-labeldir noniid-#label1 noniid-#label2 noniid-#label3 iid-diff-quantity
      do
        python ../experiments.py --model=simple-cnn \
          --dataset=$data \
          --alg=$alg \
          --lr=0.01 \
          --batch-size=64 \
          --epochs=$e \
          --n_parties=10 \
          --rho=0.9 \
          --mu=0.01 \
          --comm_round=50 \
          --partition=$par \
          --beta=0.5\
          --device='cuda:0'\
          --datadir='../data/' \
          --logdir='../logs/' \
          --noise=0\
          --sample=0\
          --init_seed=0
      done
      python ../experiments.py --model=simple-cnn \
          --dataset=$data \
          --alg=$alg \
          --lr=0.01 \
          --batch-size=64 \
          --epochs=$e \
          --n_parties=10 \
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
	
