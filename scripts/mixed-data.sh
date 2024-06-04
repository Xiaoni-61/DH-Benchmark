for par in noniid-labeldir iid-diff-quantity
do
  for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
  do
    python ../experiments.py --model=simple-cnn \
      --dataset=cifar10 \
      --alg=$alg \
      --lr=0.01 \
      --batch-size=64 \
      --epochs=10 \
      --n_parties=10 \
      --rho=0.9 \
      --mu=0.01 \
      --comm_round=50 \
      --partition=$par \
      --beta=0.5\
      --device='cuda:0'\
      --datadir='../data/' \
      --logdir='../logs/' \
      --noise=0.1\
      --sample=0\
      --init_seed=0
  done
done

	
