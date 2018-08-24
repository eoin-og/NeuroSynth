import numpy as np
from scipy import signal

import torch
import torch.utils.data
from torch.autograd import grad


def gradient_penalty(x, y):
	gradients = grad(outputs=y, inputs=x,
					 create_graph=True, retain_graph=True, only_inputs=True,
					 grad_outputs=torch.ones(y.size()))[0]
	return ((gradients.norm(2, dim=1) - 1)**2).mean()

def apply_envelope(sample, a, d, s, r):
	sample[:a] *= np.linspace(0, 1.0, a)
	sample[a:a+d] *= np.linspace(1.0, s, d)
	sample[a+d:-r] *= s
	sample[-r:] *= np.linspace(s, 0.0, r)
	return sample

def create_multi_onehot(labels, dims, batch_size):
	onehots = []
	for i in range(labels.size(1)):
		onehot = torch.zeros(batch_size, dims[i])
		onehot[np.arange(batch_size), labels[:, i].long()] = 1
		onehots.append(onehot)
	return torch.cat(onehots, dim=1)

def create_dataset(seq_length=64, dataset_size=2048, batch_size=64, att=None, sus=None, dec=None):
	freq_hz = seq_length // 10
	noise_factor = 0.05
	wave_funcs = [np.sin, signal.sawtooth, signal.square]

	samples_list = []
	for _ in range(dataset_size):
		x = 2 * np.pi * np.arange(seq_length) * freq_hz / seq_length
		x += np.random.uniform(-noise_factor*4, noise_factor*4, size=seq_length)
		wave_func = np.random.choice(wave_funcs)
		basic = wave_func(x) + np.random.uniform(-noise_factor, noise_factor, seq_length)

		sample = basic.copy()

		a_class = np.random.choice(len(att))
		a = int(att[a_class])

		s_class = np.random.choice(len(sus))
		s = sus[s_class]

		d_class = np.random.choice(len(dec))
		d = int(dec[d_class])

		sample[:a] *= np.linspace(0, 1.0, a)
		sample[a:-d] *= s
		sample[-d:] *= np.linspace(s, 0.0, d)

		class_labels = [a_class, s_class, d_class]
		
		samples_list.append((np.vstack((sample, basic, np.arange(seq_length))), class_labels))
		
	samples, class_labels = list(zip(*samples_list))
	
	data_tensor = torch.FloatTensor(samples).view(dataset_size, 3, seq_length)
	target_tensor = torch.LongTensor(class_labels)
	oh_target_tensor = create_multi_onehot(target_tensor, [len(att), len(sus), len(dec)], dataset_size)
	
	dataset = torch.utils.data.TensorDataset(data_tensor, oh_target_tensor)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)