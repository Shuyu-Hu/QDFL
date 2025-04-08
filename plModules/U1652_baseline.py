from shutil import which
from typing import Union, Optional, Callable, Any
import pytorch_lightning as pl
import torch
import numpy as np
import utils.grad_optim as ugo
from torch.optim import lr_scheduler, Optimizer
from model import get_backbone_components
from utils import print_nb_params
import utils.cal_loss as Loss
from utils import loss_func
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

from utils.cal_loss import cal_mmd_loss, cal_mse_loss

torch.set_float32_matmul_precision('medium')


class U1652_model(pl.LightningModule):
    def __init__(self,
                 backbone_name,
                 backbone_configs,
                 components_name,
                 components_configs,
                 views=2,
                 optimizer=None,
                 grad_optimizer_name=None,
                 loss_config=None,
                 lr=0.01,
                 lr_sched='linear',
                 lr_sched_config=None,
                 weight_decay=0.05,
                 momentum=0.9,
                 warmpup_steps=0
                 ):
        super().__init__()
        if loss_config is None:
            loss_config = {
                'margin': 0.3,
                'hard_factor': 0.0,
                'kernel_mul': 2,
                'kernel_num': 5
            }
        self.automatic_optimization = not grad_optimizer_name
        self.backbone_name = backbone_name
        self.backbone_configs = backbone_configs
        self.components_name = components_name
        self.components_configs = components_configs
        self.backbone = get_backbone_components.get_backbone(self.backbone_name, self.backbone_configs)
        self.components = get_backbone_components.get_components(self.components_name, self.components_configs)
        self.num_classes = components_configs['num_classes']
        self.views = views
        self.optimizer = optimizer
        self.grad_optimizer_name = grad_optimizer_name
        self.lr = lr
        self.lr_sched = lr_sched
        self.lr_sched_config = lr_sched_config
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.save_hyperparameters()
        self.CEloss = nn.CrossEntropyLoss()

        self.which_contrast_loss = loss_config.get('which_contrast_loss', 'Triplet')
        self.add_KLloss = loss_config.get('add_KLloss', False)
        self.add_MMDloss = loss_config.get('add_MMDloss', False)
        self.add_MSEloss = loss_config.get('add_MSEloss', False)


        if self.add_KLloss:
            self.KLloss = nn.KLDivLoss(reduction='batchmean')
        if self.add_MMDloss:
            self.MMDloss = loss_func.MMD_loss(kernel_mul=loss_config['kernel_mul'],
                                              kernel_num=loss_config['kernel_num'])
        if self.add_MSEloss:
            self.MSEloss = nn.MSELoss()

        # self.CEloss = loss_func.LS_CE_loss()
        if 'triplet' in self.which_contrast_loss.lower():
            self.Tripletloss = loss_func.Tripletloss(margin=loss_config['margin'],
                                                     hard_factor=loss_config['hard_factor'])
        elif 'multi' in self.which_contrast_loss.lower():
            self.MSloss = losses.MultiSimilarityLoss(alpha=5.0, beta=100, base=0.1, distance=DotProductSimilarity())
            self.MSminer = miners.MultiSimilarityMiner(epsilon=loss_config['margin'], distance=CosineSimilarity())
            # self.MSminer = None
        elif 'infonce' in self.which_contrast_loss.lower():
            self.InfoNCEloss = loss_func.InfoNCE_loss(nn.CrossEntropyLoss(label_smoothing=0.1), self.device)
        # self.tau = nn.Parameter(torch.ones([1]) * np.log(1 / 0.07))
        self.preds, self.preds2, self.preds3, self.labels_sat, self.labels_street, self.labels_drone = [], [], [], [], [], []

    def forward(self, x):
        # [B,patch_size + 1, dim]
        x = self.backbone(x)
        x = self.components(x)
        return x

    def configure_optimizers(self):

        scaler = torch.cuda.amp.GradScaler()

        ignored_params = set(map(id, self.backbone.parameters()))

        base_params = filter(lambda p: id(p) in ignored_params, self.parameters())
        extra_params = filter(lambda p: id(p) not in ignored_params, self.parameters())

        if self.optimizer.lower() == 'sgd':
            base_optimizer = torch.optim.SGD([
                {'params': base_params, 'lr': 0.3 * self.lr},
                {'params': extra_params, 'lr': self.lr}
            ],
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=True)
        elif self.optimizer.lower() == 'adamw':
            base_optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': 0.3 * self.lr},
                {'params': extra_params, 'lr': self.lr}
            ],
                weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            base_optimizer = torch.optim.Adam([
                {'params': base_params, 'lr': 0.3 * self.lr},
                {'params': extra_params, 'lr': self.lr}
            ])
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        if self.grad_optimizer_name not in [None, False]:
            if 'pcgrad' in self.grad_optimizer_name.lower():
                self.grad_optimizer = ugo.PCGradAMP(num_tasks=3, optimizer=base_optimizer, scaler=scaler,
                                                reduction='sum', cpu_offload=False)
                optimizer = self.grad_optimizer.optimizer
            elif 'gradvac' in self.grad_optimizer_name.lower():
                self.grad_optimizer = ugo.GradVacAMP(num_tasks=3, optimizer=base_optimizer, scaler = scaler,
                                                     beta = 1e-2, reduction='sum', cpu_offload = False)
                optimizer = self.grad_optimizer.optimizer
            else:
                raise KeyError('wrong grad_optimizer name or grad_optimizer method not implemented')
        else:
            optimizer = base_optimizer

        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_config['milestones'],
                                                 gamma=self.lr_sched_config['gamma'])
        elif self.lr_sched.lower() == 'steplr':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_sched_config['step_size'],
                                            gamma=self.lr_sched_config['gamma'], last_epoch=-1)
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_config['T_max'])
        elif self.lr_sched.lower() == 'cosine_schedule_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_training_steps=self.lr_sched_config['train_steps'],
                                                        num_warmup_steps=self.lr_sched_config['warmup_steps'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_config['start_factor'],
                end_factor=self.lr_sched_config['end_factor'],
                total_iters=self.lr_sched_config['total_iters']
            )
        else:
            raise ValueError(f'Scheduler {self.lr_schedulers()} has not been added to "configure_scheduler()"')
        return [optimizer], [scheduler]

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[list],
            optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        # prevent the warmup strategy uses two times
        if self.trainer.global_step < self.warmpup_steps and not 'with_warmup' in self.lr_sched.lower():
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                # pg['lr'] = lr_scale * self.lr
                pg['lr'] = lr_scale * self.lr
                # self.log(f'current_lr', pg['lr'], prog_bar=True, logger=True)

        # Optimizer step
        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()

        for pg in optimizer.param_groups:
            self.log(f'current_lr', pg['lr'], prog_bar=True, logger=True)

    def on_train_epoch_start(self) -> None:
        (self.preds, self.preds2, self.labels_sat,
         self.labels_street, self.labels_drone) = [], [], [], [], []

    def training_step(self, batch, batch_idx):
        ((inputs_sat, labels_sat), (inputs_street, labels_street),
         (inputs_drone, labels_drone)) = batch
        self.labels_sat.append(labels_sat)
        self.labels_street.append(labels_street)
        self.labels_drone.append(labels_drone)
        out_sat = self(inputs_sat)
        out_street = None
        out_drone = self(inputs_drone)
        contrast = False
        if out_sat[1] is not None and out_drone[1] is not None:
            contrast = True
        if self.views == 3:
            out_street = self(inputs_street)
        # out_sat/out_drone[1] indicates the output feature/descriptor
        if contrast:
            if 'triplet' in self.which_contrast_loss.lower():
                contrast_loss = Loss.cal_triplet_loss(out_sat[1], out_drone[1], labels_sat, self.Tripletloss)
                contrast_loss += Loss.cal_triplet_loss(out_drone[1], out_sat[1], labels_sat, self.Tripletloss)
            elif 'multi' in self.which_contrast_loss.lower():
                contrast_loss = Loss.cal_MS_loss(out_sat[1], out_drone[1], labels_sat, self.MSloss, self.MSminer)
                contrast_loss += Loss.cal_MS_loss(out_drone[1], out_sat[1], labels_sat, self.MSloss, self.MSminer)
            elif 'infonce' in self.which_contrast_loss.lower():
                contrast_loss = self.InfoNCEloss(out_sat[1], out_drone[1], self.tau)
                contrast_loss += self.InfoNCEloss(out_drone[1], out_sat[1], self.tau)
        else:
            contrast_loss = 0

        if self.add_MMDloss:
            # In order to realise domain generalization, extract the fusion_feature to calculate mmd loss
            mmdloss = 0.5*cal_mmd_loss(out_drone[2], out_sat[2], self.MMDloss) + 0.5*cal_mmd_loss(out_sat[2], out_drone[2], self.MMDloss)
        else:
            mmdloss = torch.tensor((0))

        if self.add_MSEloss:
            # In order to realise domain generalization, extract the fusion_feature to calculate mmd loss
            mseloss = cal_mse_loss(out_drone[2], out_sat[2], self.MSEloss)
        else:
            mseloss = torch.tensor((0))

        # out_sat/out_drone[0] indicates the output cls digit
        out_sat, out_drone = out_sat[0], out_drone[0]
        if not isinstance(out_sat, list):
            out_sat, out_drone = [out_sat[0]], [out_drone[0]]
        preds = []
        preds2 = []
        for out, out2 in zip(out_sat, out_drone):
            preds.append(torch.max(out, 1)[1])
            preds2.append(torch.max(out2, 1)[1])
        self.preds.append(preds)
        self.preds2.append(preds2)
        if self.views == 3:
            out_street = out_street[0]
            if not isinstance(out_street, list):
                out_street = [out_street]
            preds3 = []
            for out3 in out_street:
                self.preds3.append(torch.max(out3, 1)[1])
            self.preds3.append(preds3)

            cls_loss = Loss.cal_loss(out_sat, labels_sat, self.CEloss) + Loss.cal_loss(out_drone, labels_drone,
                                                                                       self.CEloss) + Loss.cal_loss(
                out_street, labels_street, self.CEloss)
        else:
            cls_loss = (Loss.cal_loss(out_sat, labels_sat, self.CEloss) + Loss.cal_loss(out_drone, labels_drone,
                                                                                        self.CEloss))


            if self.add_KLloss:
                # input: drone view, target: satellite view
                # this kl objective function is deployed to align descriptors which generated by different view points,
                # to make those vector having viewpoint invariant
                # klloss = 0.5*Loss.cal_kl_loss(out_drone, out_sat, self.KLloss)
                # klloss += 0.5*Loss.cal_kl_loss(out_sat, out_drone, self.KLloss)
                klloss = 0.5*Loss.cal_js_loss(out_drone, out_sat, self.KLloss)
                klloss += 0.5*Loss.cal_js_loss(out_sat, out_drone, self.KLloss)
            else:
                klloss = torch.tensor((0))

        # 累积损失和准确率
        self.log('cls_loss', cls_loss.item(), prog_bar=True, logger=True)
        if contrast:
            self.log(f'{self.which_contrast_loss}', contrast_loss.item(), prog_bar=True, logger=True)
        if self.add_KLloss:
            self.log('kl_loss', klloss.item(), prog_bar=True, logger=True)
        if self.add_MMDloss:
            self.log('MMDloss', mmdloss.item(), prog_bar=True, logger=True)
        if self.add_MSEloss:
            self.log('MSEloss', mseloss.item(), prog_bar=True, logger=True)


        if self.grad_optimizer_name:
            loss = [cls_loss, contrast_loss, klloss]
            self.grad_optimizer.zero_grad()
            self.grad_optimizer.backward(loss)

            self.grad_optimizer.step()
            self.log('avg_loss', sum(loss), prog_bar=True, logger=True)
            return {'loss': sum(loss)}
        else:
            loss = cls_loss + contrast_loss + klloss + 0.2*mmdloss + mseloss
            self.log('avg_loss', loss.item(), prog_bar=True, logger=True)
            return {'loss': loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        else:
            lr_schedulers.step()

        self.running_corrects_sat = self.calculate_accuracy(self.preds, self.labels_sat)
        try:
            self.running_corrects_street = self.calculate_accuracy(self.preds3, self.labels_street)
            self.running_corrects_drone = self.calculate_accuracy(self.preds2, self.labels_drone)
        except:
            self.running_corrects_drone = self.calculate_accuracy(self.preds2, self.labels_drone)

        if self.views==3:
            self.log('street_train_acc', self.running_corrects_street, prog_bar=True, logger=True)
        self.log('sat_train_acc', self.running_corrects_sat, prog_bar=True, logger=True)
        self.log('drone_train_acc', self.running_corrects_drone, prog_bar=True, logger=True)
        self.running_corrects_sat = 0
        self.running_corrects_street = 0
        self.running_corrects_drone = 0

    def calculate_accuracy(self, preds, labels):
        corrects = 0
        if not isinstance(preds, list):
            preds = [preds]
        for pred_list, label in zip(preds, labels):
            correct = 0
            for pred in pred_list:
                correct += float(torch.sum(pred == label).item())
            corrects += correct / len(pred_list)
        # return corrects/self.num_classes
        # len(labels)*len(labels[0]) indicates the total epoch images
        return corrects / (len(labels) * len(labels[0]))

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


def main():
    H = 256
    x = torch.randn(20, 3, H, H)

    options = {
        'backbone_name': "VIT-S",
        "backbone_configs": {
            'img_size': (256, 256),
        },
        'num_classes': 701,
        "components_name": "FSRA",
        "components_configs": {
            'num_classes': 701,
            'num_blocks': 3,
            'feature_dim': 768,
            'return_f': 0.3
        },
        'views': 2,
        'optimizer': 'sgd',
        'loss_config': {
            'margin': 0.3,
            'hard_factor': 0.0
        },
        'lr': 0.01,
        'lr_sched': 'multistep',
        'lr_sched_config': {
            'milestones': [70, 110],
            'gamma': 0.1
        },
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'warmpup_steps': 0
    }

    m = U1652_model(**options)
    z = m(x)
    # z = torch.stack(z,dim=0)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    if isinstance(z, tuple):
        print(f'Training Output shape is cls:{torch.stack(z[0], dim=0).shape},feature:{torch.stack(z[1], dim=0).shape}')
    else:
        print(f'Training Output shape is {torch.stack(z, dim=0).shape}')

    m.train(False)
    z = m(x)
    if isinstance(z, tuple):
        print(
            f'Evaluating Output shape is cls:{torch.stack(z[0], dim=0).shape},feature:{torch.stack(z[1], dim=0).shape}')
    else:
        print(f'Evaluating Output shape is {z.shape}')


if __name__ == "__main__":
    main()
