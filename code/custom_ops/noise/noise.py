import sys

sys.path.insert(0, './')
from torch import nn
from torch.autograd import Function
import utils.neural_texture_helper as utils_nt
import noise_cuda
import torch
from torch.autograd import gradcheck


class NoiseFunction(Function):
    @staticmethod
    def forward(ctx, position, seed):
        ctx.save_for_backward(position, seed)
        noise = noise_cuda.forward(position, seed)
        return noise

    @staticmethod
    def backward(ctx, grad_noise):
        position, seed = ctx.saved_tensors
        d_position_bilinear = noise_cuda.backward(position, seed)

        d_position = torch.stack([torch.zeros_like(d_position_bilinear), d_position_bilinear], dim=0)

        return grad_noise.unsqueeze(2) * d_position, None


class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

    def forward(self, position, seed):

        noise = NoiseFunction.apply(position.contiguous(), seed.contiguous())

        return noise


## TEST ##
if __name__ == "__main__":

    torch.manual_seed(3)

    batch_size = 1
    device = 'cuda'
    bs = 1
    octaves = 1
    dim = 2
    h = 1
    w = 1
    texture_channels = 1

    position = utils_nt.get_position((h, w), dim, device, bs) + 0.8
    position.requires_grad = True

    bs, dim, h, w = position.size()
    seed = torch.rand((bs, octaves, texture_channels), device=device)
    t_coeff = torch.rand((bs, octaves * dim * dim, 1, 1), device=device)
    t_coeff = torch.ones_like(t_coeff) + 1
    t_coeff.requires_grad = True

    noise_sampler = Noise().to('cuda')

    position = position.unsqueeze(1).expand(bs, octaves, dim, h, w)
    position = position.permute(0, 1, 3, 4, 2)

    position = utils_nt.transform_coord(position, t_coeff, dim)

    # multiply with 2**i to initializate octaves
    octave_factor = torch.arange(0, octaves, device=device)
    octave_factor = octave_factor.reshape(1, octaves, 1, 1, 1)
    octave_factor = octave_factor.expand(1, octaves, 1, 1, dim)
    octave_factor = torch.pow(2, octave_factor)
    position = position * octave_factor

    bs, octaves, h, w, dim = position.size()
    texture_channels = seed.shape[2]

    # position
    position = position.unsqueeze(2).expand(bs, octaves, texture_channels, h, w, dim)
    seed = seed.unsqueeze(-1).unsqueeze(-1).expand(bs, octaves, texture_channels, h, w)
    position = position.reshape(bs * octaves * texture_channels * h * w, dim)
    seed = seed.reshape(bs * octaves * texture_channels * h * w)

    noise = noise_sampler(position, seed)
    noise = noise.reshape(-1, bs, octaves, texture_channels, h, w)

    noise.mean().backward()
