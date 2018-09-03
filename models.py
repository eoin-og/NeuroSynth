import torch
import torch.nn as nn
import torch.nn.functional as F 

class ResBlock(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.main = nn.Sequential(
			nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm1d(dim, affine=True),
			nn.ReLU(inplace=True),

			nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm1d(dim, affine=True)
			)

	def forward(self, x):
		return self.main(x)


class Discriminator(nn.Module):
	
	def __init__(self, num_dim, seq_length, conv_dim):
		super().__init__()
		num_layer = 3
		final_kernel_size = seq_length // (2**num_layer)

		self.conv1 = nn.Conv1d(3, conv_dim, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv1d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv1d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1)

		self.in1 = nn.InstanceNorm1d(conv_dim, affine=True)
		self.in2 = nn.InstanceNorm1d(conv_dim*2, affine=True)
		self.in3 = nn.InstanceNorm1d(conv_dim*4, affine=True)

		self.cls_output = nn.Conv1d(conv_dim*4, num_dim, kernel_size=final_kernel_size, bias=False)
		self.src_output = nn.Conv1d(conv_dim*4, 1, kernel_size=final_kernel_size, bias=False)

	
	def forward(self, x):

		x = F.leaky_relu(self.in1(self.conv1(x)), 0.01)
		x = F.leaky_relu(self.in2(self.conv2(x)), 0.01)
		x = F.leaky_relu(self.in3(self.conv3(x)), 0.01)

		src_output = F.sigmoid(self.src_output(x))
		cls_output = self.cls_output(x)

		return src_output.view(src_output.size(0), src_output.size(1)), cls_output.view(cls_output.size(0), cls_output.size(1))


class Generator(nn.Module):

	def __init__(self, num_dim, conv_dim):
		super().__init__()

		self.conv1 = nn.Conv1d(2+num_dim, conv_dim, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv1d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1)

		self.res1 = ResBlock(conv_dim*2)
		self.res2 = ResBlock(conv_dim*2)

		self.tcnv1 = nn.ConvTranspose1d(conv_dim*2, conv_dim, kernel_size=4, stride=2, padding=1)
		self.tcnv2 = nn.ConvTranspose1d(conv_dim, 1, kernel_size=4, stride=2, padding=1)

		self.in1 = nn.InstanceNorm1d(conv_dim, affine=True)
		self.in2 = nn.InstanceNorm1d(conv_dim*2, affine=True)
		self.in3 = nn.InstanceNorm1d(conv_dim, affine=True)

	
	def forward(self, x):

		x = F.leaky_relu(self.in1(self.conv1(x)), 0.01)
		x = F.leaky_relu(self.in2(self.conv2(x)), 0.01)

		x = self.res1(x)
		x = self.res2(x)

		x = F.leaky_relu(self.in3(self.tcnv1(x)), 0.01)
		x = F.tanh(self.tcnv2(x))

		return x
