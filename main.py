import argparse
import gan


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Training script for NeuroSynth GAN.')
	parser.add_argument('--num_epoch', type=int, default=2000, metavar='N',
						help='# of training epochs')
	parser.add_argument('--dsize', type=int, default=2048, metavar='N',
						help='# of training examples')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='training batch size')
	parser.add_argument('--seq-length', type=int, default=128, metavar='N',
						help='length of generated samples')
	parser.add_argument('--conv-dim', type=int, default=8, metavar='N',
						help='# of dimensions in first conv layer')
	parser.add_argument('--print-interval', type=int, default=10, metavar='N',
						help='After how many epochs should print results/save model')
	parser.add_argument('--sustains', type=list, default=[1.0, 0.7, 0.4], metavar='N', 
						help='sustain levels')
	parser.add_argument('--attacks', type=list, default=[0.0, 0.2, 0.4], metavar='N', 
						help='attack levels (amount of sample that should be rising)')
	parser.add_argument('--releases', type=list, default=[0.0, 0.2, 0.4], metavar='N', 
						help='release levels (amount of sample that should be falling)')
	args = parser.parse_args()

	attacks = [int(a*args.seq_length) + 1 for a in args.attacks]
	releases = [int(a*args.seq_length) + 1 for a in args.releases]
	variables = [args.sustains, attacks, releases]
	
	gan.train(num_epoch=args.num_epoch,
			  dsize=args.dsize,
			  batch_size=args.batch_size,
			  print_interval=args.print_interval,
			  variables=variables,
			  cf=args.conv_dim,
			  seq_length=args.seq_length)