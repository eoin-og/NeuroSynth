import random
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import models
import utils

np.set_printoptions(precision=3)


def train(num_epoch=500, dsize=2048, batch_size=64, seq_length=128, cf=8, print_interval=10, variables=None):

	num_dim = sum(len(v) for v in variables)
	sustains, attacks, decays = variables


	G = models.Generator(num_dim, cf)
	D = models.Discriminator(num_dim, seq_length, cf)

	D_optimizer = torch.optim.Adam(D.parameters())
	G_optimizer = torch.optim.Adam(G.parameters())

	loss_function = F.binary_cross_entropy_with_logits

	dataloader = utils.create_dataset(seq_length, dsize, batch_size, sus=sustains, att=attacks, dec=decays)

	for epoch in range(num_epoch):
		for sample, label in dataloader:

			input_labels = label.view(batch_size, num_dim, 1).repeat(1, 1, seq_length)
			main_input = torch.cat([sample, input_labels], dim=1)
			real_sample = sample[:, 0, :].clone()


			############
			# D - real #
			############		
			d_real_src_pred, d_real_cls_pred = D(main_input[:, :3, :])
			d_real_loss = -torch.mean(d_real_src_pred)
			d_real_cls_loss = loss_function(d_real_cls_pred, label)

			D_optimizer.zero_grad()
			d_real_loss = d_real_loss + d_real_cls_loss
			d_real_loss.backward()
			D_optimizer.step()

			############
			# G - fake #
			############
			gen_sample = G(main_input[:, 1:, :])
			main_input = torch.cat([gen_sample, main_input[:, 1:, :]], dim=1)
			d_fake_src_pred, _ = D(main_input[:, :3, :].detach())
			d_fake_loss = torch.mean(d_fake_src_pred)

			D_optimizer.zero_grad()
			d_fake_loss.backward()
			D_optimizer.step()

			############
			#    GP    #
			############
			alpha = torch.rand(batch_size, 1)
			interpolates = alpha*real_sample.squeeze().data + (1 - alpha)*gen_sample.squeeze().data
			interpolates = interpolates.view(batch_size, 1, seq_length)
			interpolates = interpolates.requires_grad_(True)
			interp_input = torch.cat([interpolates, main_input[:, 1:, :]], dim=1)

			gp_pred, _ = D(interp_input[:, :3, :])
			gp_loss = utils.gradient_penalty(x=interp_input, y=gp_pred)

			D_optimizer.zero_grad()
			gp_loss.backward(retain_graph=True)
			D_optimizer.step()

			############
			#   Gen    #
			############
			gen_sample = G(main_input[:, 1:, :])
			main_input = torch.cat([gen_sample, main_input[:, 1:, :]], dim=1)
			g_src_pred, g_cls_pred = D(main_input[:, :3, :])
			g_src_loss = -torch.mean(g_src_pred)
			g_cls_loss = loss_function(g_cls_pred, label)

			G_optimizer.zero_grad() ; D_optimizer.zero_grad()
			g_loss = g_src_loss + g_cls_loss
			g_loss.backward()
			G_optimizer.step()
			
		
		if epoch % print_interval == 0:
			print('\n----- ', epoch, '-----')
			print('D real: ', d_real_src_pred.data.mean().numpy(), '--- class: ', d_real_cls_loss.data.numpy(), 
					'--- gp: ', gp_loss.data.numpy())
			print('G loss: ', g_src_pred.data.mean().numpy(), '--- class: ', g_cls_loss.data.numpy())

			i = np.random.randint(0, batch_size - 1)
			gs = gen_sample[i, :].squeeze().data.numpy()
			rs = real_sample[i, :].data.numpy()
			lab = label[i].data.numpy()

			print('label: ', lab)
			print('real pred : ', d_real_cls_pred[i].data.numpy())
			print('fake pred : ', g_cls_pred[i].data.numpy())
			
			plt.plot(gs, 'r')
			plt.plot(rs, 'g')
			plt.title('label: {}'.format(lab))
			plt.savefig('output_data/gen_sample_epoch_{}.png'.format(epoch))
			plt.clf()

			#filepath = 'models/gen/G_{}.pth.tar'.format(epoch)
			#torch.save(G.state_dict(), filepath)

			#filepath = 'models/disc/D_{}.pth.tar'.format(epoch)
			#torch.save(D.state_dict(), filepath)

