for init_seed in 0 1 2
do
	for dataset in cifar10 mnist fmnist svhn
	do
		for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
		do
			python ../experiments.py --model=simple-cnn \
				--dataset=$dataset \
				--alg=$alg \
				--lr=0.01 \
				--batch-size=64 \
				--epochs=10 \
				--n_parties=10 \
				--rho=0.9 \
				--comm_round=50 \
				--partition=homo \
				--beta=0.5\
				--device='cuda:0'\
				--datadir='../data/' \
				--logdir='../logs/' \
				--noise=0.1\
				--init_seed=$init_seed
		done
	done
done


