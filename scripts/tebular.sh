for init_seed in 0 1 2
do
	for partition in noniid-labeldir noniid-#label1 iid-diff-quantity homo
	do
		for dataset in rcv1
		do
			for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
			do
				python ../experiments.py --model=mlp \
					--dataset=$dataset \
					--alg=$alg \
					--lr=0.1 \
					--batch-size=64 \
					--epochs=10 \
					--n_parties=10 \
					--rho=0.9 \
					--comm_round=50 \
					--partition=$partition \
					--beta=0.5\
					--device='cuda:0'\
					--datadir='./data/' \
					--logdir='./logs/' \
					--noise=0\
					--init_seed=$init_seed
			done
			
		done
		for dataset in covtype a9a
		do
			for alg in pFedJS pFedgraph FedCollab pFedSV RACE CE
			do
				python ../experiments.py --model=mlp \
					--dataset=$dataset \
					--alg=$alg \
					--lr=0.01 \
					--batch-size=64 \
					--epochs=10 \
					--n_parties=10 \
					--rho=0.9 \
					--comm_round=50 \
					--partition=$partition \
					--beta=0.5\
					--device='cuda:0'\
					--datadir='../data/' \
					--logdir='../logs/' \
					--noise=0\
					--init_seed=$init_seed
			done

		done
	done
done
