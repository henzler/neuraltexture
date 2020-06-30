import torch
from systems.s_core import CoreSystem
import utils.io as io
import utils.utils as utils
from pathlib import Path
import numpy as np
from custom_ops.noise.noise import Noise
import utils.neural_texture_helper as utils_nt
import copy
import json
import kornia


class SystemNeuralTexture(CoreSystem):

    def __init__(self, param):
        super().__init__(param)

        self.image_res = self.p.image.image_res

        if self.p.dataset.use_single != -1:
            weight_size = self.p.texture.e + self.p.texture.t
            weights = torch.Tensor(1, weight_size, 1, 1).uniform_(-1.0, 1.0).to(self.p.device)
            self.register_parameter('weights', torch.nn.Parameter(weights))
            self.optimizers[0].add_param_group({'params': self.weights})
        else:
            self.model_encoder_texture = self.models[0]

        self.model_texture_mlp = self.models[1]
        self.loss_style_type = self.get_loss_type(self.p_system.block_main.loss_params.style_type)

        self.vgg_features = utils_nt.VGGFeatures()
        self.gram_matrix = utils_nt.GramMatrix()

        self.identity_matrix = torch.nn.init.eye_(torch.empty(self.p.dim, self.p.dim, device=self.p.device))
        self.sample_pos = utils_nt.get_position((self.image_res, self.image_res), self.p.dim, self.p.device, self.p.train.bs)

        self.noise_sampler = Noise().to(self.p.device)

        # test config
        self.tmp_image_gt = None
        self.n_seeds = 3
        self.strip_length = 10
        self.n_zooms = 10
        self.n_interpolations = 5

        # re-seed in case of different numbers of parameters in networks
        self.seed()

    def parse_weights(self, image_gt):

        if self.p.dataset.use_single != -1:
            weights = self.weights.expand(self.p.train.bs, self.p.texture.e + self.p.texture.t, 1, 1)
            weights_bottleneck = None
        else:
            weights, weights_bottleneck = self.model_encoder_texture(image_gt)
            weights = weights.unsqueeze(-1).unsqueeze(-1)

        return weights, weights_bottleneck

    def forward(self, weights, position, seed):

        _, _, h, w = position.size()
        bs, _, w_h, w_w = weights.size()

        transform_coeff, z_encoding = torch.split(weights, [self.p.texture.t, self.p.texture.e], dim=1)

        if z_encoding.shape[2] == 1:
            z_encoding = z_encoding.view(bs, self.p.texture.e, 1, 1)
            z_encoding = z_encoding.expand(bs, self.p.texture.e, self.image_res, self.image_res)

        position = position.unsqueeze(1).expand(bs, self.p.noise.octaves, self.p.dim, h, w)
        position = position.permute(0, 1, 3, 4, 2)

        position = utils_nt.transform_coord(position, transform_coeff, self.p.dim)

        # multiply with 2**i to initializate octaves
        octave_factor = torch.arange(0, self.p.noise.octaves, device=self.p.device)
        octave_factor = octave_factor.reshape(1, self.p.noise.octaves, 1, 1, 1)
        octave_factor = octave_factor.expand(1, self.p.noise.octaves, 1, 1, self.p.dim)
        octave_factor = torch.pow(2, octave_factor)
        position = position * octave_factor

        # position
        position = position.unsqueeze(2).expand(bs, self.p.noise.octaves, self.p.texture.channels, h, w, self.p.dim)
        seed = seed.unsqueeze(-1).unsqueeze(-1).expand(bs, self.p.noise.octaves, self.p.texture.channels, h, w)
        position = position.reshape(bs * self.p.noise.octaves * self.p.texture.channels * h * w, self.p.dim)
        seed = seed.reshape(bs * self.p.noise.octaves * self.p.texture.channels * h * w)

        noise = self.noise_sampler(position, seed)
        noise = noise.reshape(-1, bs, self.p.noise.octaves, self.p.texture.channels, h, w)
        noise = noise.permute(1, 0, 2, 3, 4, 5)
        noise = noise.reshape(bs, self.p.noise.octaves * self.p.texture.channels * 2, self.p.image.image_res, self.p.image.image_res)

        input_mlp = torch.cat([z_encoding, noise], dim=1)
        image_out = self.model_texture_mlp(input_mlp)
        image_out = torch.tanh(image_out)

        return image_out

    def training_step(self, batch, batch_nb):

        image_gt, image_out = self.forward_step(batch)

        loss = self.get_loss(image_gt, image_out, 'train')
        self.log(image_gt, image_out, 'train')

        return loss

    def validation_step(self, batch, batch_nb):

        image_gt, image_out = self.forward_step(batch)
        loss = self.get_loss(image_gt, image_out, 'val')
        self.log(image_gt, image_out, 'val', force=True)

        return loss

    def process_input(self, batch):
        return utils.unsigned_to_signed(batch)

    def forward_step(self, batch):

        seed = torch.rand((self.p.train.bs, self.p.noise.octaves, self.p.texture.channels), device=self.p.device)

        image_gt = self.process_input(batch)

        # speed up 2D case
        if self.p.dim == 2:
            sample_pos = self.sample_pos
        else:
            sample_pos = utils_nt.get_position((self.image_res, self.image_res), self.p.dim, self.p.device, self.p.train.bs)

        weights, _ = self.parse_weights(image_gt)
        image_out = self.forward(weights, sample_pos, seed)

        return image_gt, image_out

    def get_loss(self, image_gt_cropped, image_out_shaded, mode='train'):

        loss_style = torch.tensor(0.0, device=self.p.device)

        vgg_features_out = self.vgg_features(utils.signed_to_unsigned(image_out_shaded))
        vgg_features_gt = self.vgg_features(utils.signed_to_unsigned(image_gt_cropped))
        gram_matrices_gt = list(map(self.gram_matrix, vgg_features_gt))
        gram_matrices_out = list(map(self.gram_matrix, vgg_features_out))

        for gram_matrix_gt, gram_matrix_out in zip(gram_matrices_gt, gram_matrices_out):
            loss_style += self.p_system.block_main.loss_params.style_weight * self.loss_style_type(gram_matrix_out, gram_matrix_gt)

        if mode == 'train':
            losses = {'loss': loss_style, 'progress_bar': {'loss': loss_style}, 'log': {'loss': loss_style}}
        else:
            losses = {'{}_loss'.format(mode): loss_style}

        return losses

    def log(self, image_gt, image_out, mode='train', force=False):

        with torch.no_grad():
            self.logger.log_multi_channel_image('{}_image_gt'.format(mode), utils.signed_to_unsigned(image_gt), self.global_step, force=force)
            self.logger.log_multi_channel_image('{}_image_out'.format(mode), utils.signed_to_unsigned(image_out), self.global_step, force=force)

    def test_step(self, batch, batch_nb):

        file_ending = 'jpg'

        with torch.no_grad():

            batch_data, batch_filename = batch

            filename = batch_filename[0]
            image_gt = self.process_input(batch_data)

            result_dir = Path(self.logger.results_files_dir / filename)
            result_dir.mkdir(exist_ok=True)
            weights, weights_bottleneck = self.parse_weights(image_gt)

            input_path = result_dir / 'input.{}'.format(file_ending)
            io.write_images(str(input_path), utils.signed_to_unsigned(image_gt), 1)

            position = utils_nt.get_position((self.image_res, self.image_res), self.p.dim, self.p.device, self.p.train.bs)

            style = 0.0
            diversity = 0.0
            vgg_features_out_seeds = []

            for i in range(self.n_seeds):

                seed = torch.rand((self.p.train.bs, self.p.noise.octaves, self.p.texture.channels), device=self.p.device)

                image_out = self.forward(weights, position, seed)

                vgg_features_gt = self.vgg_features(utils.signed_to_unsigned(image_gt))
                vgg_features_out = self.vgg_features(utils.signed_to_unsigned(image_out))

                vgg_features_out_seeds.append(vgg_features_out)

                for f_gt, f_out in zip(vgg_features_gt, vgg_features_out):
                    gram_f_out = self.gram_matrix(f_out)
                    gram_f_gt = self.gram_matrix(f_gt)
                    style += utils.metric_mse(gram_f_out, gram_f_gt)

                out_path = result_dir / '{}_out.{}'.format(i, file_ending)

                io.write_images(str(out_path), utils.signed_to_unsigned(image_out), 1)

                ## stripe
                image_stripe = []
                position_stripe = utils_nt.get_position((self.image_res, self.image_res), self.p.dim, self.p.device, self.p.train.bs) + 1

                for _ in range(self.strip_length):
                    position_stripe = torch.cat([position_stripe[:, 0:1] + 2.0, position_stripe[:, 1:]], dim=1)

                    image_out = self.forward(weights, position_stripe, seed).detach().cpu()
                    image_out = utils.signed_to_unsigned(image_out)
                    image_stripe.append(image_out)

                image_stripe = torch.cat(image_stripe, dim=3)
                io.write_images(str(result_dir / '{}_stripe.{}'.format(i, file_ending)), image_stripe, 1)

                ## zoom
                images_zoom = []
                position_zoom = utils_nt.get_position((self.image_res, self.image_res), self.p.dim, self.p.device, self.p.train.bs)
                for zoom in range(1, self.n_zooms + 1):
                    # sample zoom
                    image_out_zoom_out = self.forward(weights, position_zoom * (zoom / self.n_zooms), seed).detach().cpu()
                    image_out_zoom_out = utils.signed_to_unsigned(image_out_zoom_out)
                    images_zoom.append(image_out_zoom_out)

                zoom_gif = images_zoom + copy.deepcopy(images_zoom)[::-1]
                io.write_gif(result_dir / '{}_zoom.gif'.format(i), zoom_gif, 1)

            for f_stack1 in vgg_features_out_seeds:
                for f_stack2 in vgg_features_out_seeds:
                    for f_out1, f_out2 in zip(f_stack1, f_stack2):
                        diversity += utils.metric_mse(f_out1, f_out2) / self.n_seeds

            evaluation_values = (style, diversity)

            # interpolation
            interpolations = []

            if self.tmp_image_gt is not None:
                weights2, weights_bottleneck2 = self.parse_weights(self.tmp_image_gt)

                for i in range(self.n_interpolations):
                    weight_list = []
                    for z in torch.linspace((i * (1 / (self.n_interpolations))), (i + 1) * (1 / (self.n_interpolations)), steps=self.p.image.image_res):
                        weights_interpolated = weights_bottleneck * (1 - z) + z * weights_bottleneck2
                        weights_interpolated = self.model_encoder_texture.fc_final(weights_interpolated)
                        weight_list.append(weights_interpolated)

                    z_texture_interpolated = torch.stack(weight_list, dim=2).unsqueeze(-2)

                    # z_texture_interpolated = z_texture_interpolated[:, :-2]
                    latent_space = z_texture_interpolated.shape[1]
                    z_texture_interpolated = z_texture_interpolated.expand(1, latent_space, self.p.image.image_res, self.p.image.image_res)

                    position = utils_nt.get_position((self.p.image.image_res, self.p.image.image_res), self.p.dim, self.p.device, self.p.train.bs)
                    position += 1
                    position[:, 0] = position[:, 0] + (2 * i)
                    image_out_inter = self.forward(z_texture_interpolated, position, seed)

                    image_out_inter = utils.signed_to_unsigned(image_out_inter.detach().cpu())
                    interpolations.append(image_out_inter)

                image_stripe_gt = torch.cat(interpolations, dim=3)
                result_path = result_dir / 'interpolated.png'
                io.write_images(str(result_path), image_stripe_gt, 1)

            self.tmp_image_gt = image_gt

        return evaluation_values

    def test_end(self, evaluation_values):

        evaluation_keys = ('style', 'diversity')

        evaluation_values = np.array(evaluation_values)

        evaluation_json = {}

        for idx, key in enumerate(evaluation_keys):
            evaluation_json[key] = list(evaluation_values[:, idx])

        filename = Path(self.logger.results_files_dir / 'evaluation.json')

        # Writing JSON data
        with open(str(filename), 'w') as f:
            json.dump(evaluation_json, f)

        return {}
