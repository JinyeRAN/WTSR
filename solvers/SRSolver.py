import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.utils as thutil

from networks import create_model
from .base_solver import BaseSolver
from .tools import L1_Charbonnier_loss, CosineAnnealingWarmRestartsWarmup
from networks import init_weights
from utils import util

import torch.nn.functional as F


class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None
        self.scale = opt['scale']
        self.offset = opt['solver']['offset']
        self.lrsize = opt['LR_size']

        self.records = {'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': [], 'lr': []}

        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            elif loss_type == 'Cl1':
                self.criterion_pix = L1_Charbonnier_loss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                # self.optim_param = []
                if self.opt['solver']['pretrain'] == 'finetune':
                    for name, param in self.model.module.named_parameters():
                        # print(name)
                        if len(self.opt['solver']['freezelayer']) == 0:
                            param.requires_grad = False
                        else:
                            for freeze in self.opt['solver']['freezelayer']:
                                if not freeze == name.split('.')[0]:
                                    param.requires_grad = False
                                else:
                                    param.requires_grad = True
                                    # self.optim_param.append(param)
                                    print(name)
                    # self.optimizer = optim.Adam(self.model.module.fine.parameters(),
                    #                             lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
                    # self.optimizer = optim.Adam(self.optim_param,
                    #                             lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
                    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                                lr=self.train_opt['learning_rate'], weight_decay=weight_decay)

                else:
                    self.optimizer = optim.Adam(self.model.parameters(),
                                                lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            elif self.train_opt['lr_scheme'].lower() == 'cossteplr':
                self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                             T_0=10,
                                                             T_mult=2)
            elif self.train_opt['lr_scheme'].lower() == 'warmcos':
                scheduler_steplr = CosineAnnealingWarmRestarts(self.optimizer,
                                                               T_0=17,
                                                               T_mult=2,
                                                               eta_min=1e-5,
                                                               last_epoch=-1)
                self.scheduler = CosineAnnealingWarmRestartsWarmup(self.optimizer,
                                                                   multiplier=1,
                                                                   total_epoch=10,
                                                                   after_scheduler=scheduler_steplr)
            elif self.train_opt['lr_scheme'].lower() == 'warmmulti':
                scheduler_steplr = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  self.train_opt['lr_steps'],
                                                                  self.train_opt['lr_gamma'])
                self.scheduler = CosineAnnealingWarmRestartsWarmup(self.optimizer,
                                                                   multiplier=1,
                                                                   total_epoch=10,
                                                                   after_scheduler=scheduler_steplr)
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()
        # self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.print_network()

        print('===> Solver Initialized : [%s] || Use GPU : [%s]' % (self.__class__.__name__, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]' % init_type)
        init_weights(self.model, init_type)

    def feed_data(self, batch, need_HR=True):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)

        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.LR)
        loss_sbatch = self.criterion_pix(output, self.HR)
        loss_sbatch.requires_grad_(True)

        loss_sbatch.backward()

        # for stable training
        if loss_sbatch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_sbatch.item()
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_sbatch))

        self.model.eval()
        return loss_sbatch.item()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
            if self.self_ensemble and not self.is_train:
                SR = self._forward_x8(self.LR, forward_func)
            else:
                SR = forward_func(self.LR, self.lrsize, self.offset, self.scale)

            self.SR = SR

        self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix.item()

    def _forward_x8(self, x, forward_function):
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = self.Tensor(tfnp)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug, self.lrsize, self.offset, self.scale)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def _overlap_crop_forward(self, x, lrsize, offset, scale):
        F_size = lrsize - 2 * offset
        b, c, h, w = x.size()

        h_size, w_size = h // F_size, w // F_size
        h_size_pad, w_size_pad = h % F_size, w % F_size
        if h_size_pad == 0:
            h_padding = offset
            hlen = h_size
        else:
            h_padding = F_size - h_size_pad + offset
            hlen = h_size + 1

        if w_size_pad == 0:
            w_padding = offset
            wlen = w_size
        else:
            w_padding = F_size - w_size_pad + offset
            wlen = w_size + 1

        if w_padding <= w and h_padding <= h:
            p1d = (offset, w_padding, offset, h_padding)
            x = F.pad(x, p1d, mode='reflect')
        elif w_padding > w and h_padding > h:
            p1d = (0, w - 1, 0, h - 1)
            x = F.pad(x, p1d, mode='reflect')
            p2d = (offset, w_padding - w + 1, offset, h_padding - h + 1)
            x = F.pad(x, p2d, mode='reflect')
        elif w_padding < w and h_padding > h:
            p1d = (0, 0, 0, h - 1)
            x = F.pad(x, p1d, mode='reflect')
            p2d = (offset, w_padding, offset, h_padding - h + 1)
            x = F.pad(x, p2d, mode='reflect')
        elif w_padding > w and h_padding < h:
            p1d = (0, w - 1, 0, 0)
            x = F.pad(x, p1d, mode='reflect')
            p2d = (offset, w_padding - w + 1, offset, h_padding)
            x = F.pad(x, p2d, mode='reflect')
        else:
            raise

        lr_list = []
        for i in range(hlen):
            for j in range(wlen):
                if i == 0 and j == 0:
                    lr_list.append(x[:, :, i * lrsize:(i + 1) * lrsize, j * lrsize:(j + 1) * lrsize])
                elif i != 0 and j == 0:
                    lr_list.append(x[:, :, i * F_size:i * F_size + lrsize, 0:(j + 1) * lrsize])
                elif i == 0 and j != 0:
                    lr_list.append(x[:, :, 0:(i + 1) * lrsize, j * F_size:j * F_size + lrsize])
                elif i != 0 and j != 0:
                    lr_list.append(x[:, :, i * F_size:i * F_size + lrsize, j * F_size:j * F_size + lrsize])
                else:
                    print('wrong')

        sr_list = []
        for lr in lr_list:
            sr = self.model(lr)
            sr = sr[:, :, offset * scale:lrsize * scale - offset * scale,
                 offset * scale:lrsize * scale - offset * scale]
            sr_list.append(sr)

        col = []
        c = -1
        for _ in range(hlen):
            tmp = []
            for _ in range(wlen):
                c = c + 1
                tmp.append(sr_list[c])
            col.append(torch.cat(tmp, dim=-1))

        output = torch.cat(col, dim=-2)
        output = output[:, :, 0:h * scale, 0:w * scale]
        return output

    def save_checkpoint(self, epoch, is_best):
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]' % filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp', 'best_ckp'))
            torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp', 'epoch_%d_ckp' % epoch)))
            torch.save(ckp, filename.replace('last_ckp', 'epoch_%d_ckp' % epoch))

    def load(self):
        """
        load or initialize network
        """
        if self.is_train:
            if self.opt['solver']['pretrain'] == 'resume':
                model_path = self.opt['solver']['pretrained_path']
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.cur_epoch = checkpoint['epoch'] + 1
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                self.best_epoch = checkpoint['best_epoch']
                self.records = checkpoint['records']

            elif self.opt['solver']['pretrain'] == 'finetune':
                freezelayer = self.opt['solver']['freezelayer']
                self._net_init()
                model_path = self.opt['solver']['pretrained_path']
                checkpoint = torch.load(model_path)
                tmp = self.model.state_dict()
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    for freeze in freezelayer:
                        if not k.split('.')[1] == freeze:
                            state_dict[k]=v
                        else:
                            pass
                # state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k in freezelayer}
                tmp.update(state_dict)
                self.model.load_state_dict(tmp)
            else:
                self._net_init()
        else:
            model_path = self.opt['solver']['pretrained_path']
            checkpoint = torch.load(model_path)
            if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
            load_func = self.model.load_state_dict if isinstance(self.model, nn.DataParallel) else self.model.module.load_state_dict
            load_func(checkpoint)

    def get_current_visual(self, need_np=True, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_np:  out_dict['LR'], out_dict['SR'] = util.Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                     self.opt['rgb_range'])
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
            if need_np: out_dict['HR'] = util.Tensor2np([out_dict['HR']],
                                                        self.opt['rgb_range'])[0]
        return out_dict

    def save_current_visual(self, epoch, iter):
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            visuals = self.get_current_visual(need_np=False)
            visuals_list.extend([util.quantize(visuals['HR'].squeeze(0), self.opt['rgb_range']),
                                 util.quantize(visuals['SR'].squeeze(0), self.opt['rgb_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            misc.imsave(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1)),
                        visual_images)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step()

    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']

    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss']
                , 'val_loss': self.records['val_loss']
                , 'psnr': self.records['psnr']
                , 'ssim': self.records['ssim']
                , 'lr': self.records['lr']
                  },
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                             self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")
