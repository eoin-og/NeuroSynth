import gan


if __name__ == "__main__":


	num_epoch = 10000
	dsize = 2048
	batch_size = 64
	seq_length = 128
	cf = 8
	print_interval = 10

	sustains = [1.0, 0.7, 0.4]
	attacks = [1, seq_length*0.2, seq_length*0.4]
	decays =  [1, seq_length*0.2, seq_length*0.4]
	variables = [sustains, attacks, decays]

	gan.train(num_epoch=num_epoch,
			  dsize=dsize,
			  batch_size=batch_size,
			  print_interval=print_interval,
			  variables=variables,
			  cf=cf,
			  seq_length=seq_length)