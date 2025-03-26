from shutil import which
from typing import Union, Optional, Callable, Any
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import  utils.grad_optim as ugo
from torch.optim import lr_scheduler, Optimizer

import utils.loss_func
from model import get_backbone_components
from plModules import DAC_model
from plModules.Teacher_model import Teacher_model
from utils import print_nb_params, DistillKL
import utils.cal_loss as Loss
from utils import loss_func
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from plModules.U1652_baseline import U1652_model
from utils.cal_loss import cal_mmd_loss, cal_mse_loss, cal_kl_loss
from utils.commons import load_config

torch.set_float32_matmul_precision('medium')

class Teacher_Eval_Model(U1652_model):
    def __init__(self,
                 T_model_configs,
                 T_model_weights_path
                 ):
        super(Teacher_Eval_Model,self).__init__(**T_model_configs)
        # self.load_state_dict(torch.load(T_model_weights_path),strict=True)
        self.load_state_dict(torch.load(T_model_weights_path)['state_dict'],strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class Knowledge_Distill_Model(pl.LightningModule):
    def __init__(self,
                 backbone_name,
                 backbone_configs,
                 components_name,
                 components_configs,
                 T_model_configs=None,
                 T_model_weights_path=None,
                 optimizer=None,
                 grad_optimizer_name=None,
                 views=2,
                 lr=0.01,
                 lr_sched='linear',
                 lr_sched_config=None,
                 weight_decay=0.05,
                 momentum=0.9,
                 warmpup_steps=0
                 ):
        super(Knowledge_Distill_Model, self).__init__()
        if self.train():
            self.T_model = Teacher_Eval_Model(T_model_configs,T_model_weights_path)
        elif self.eval():
            self.T_model = None

        self.backbone_name = backbone_name
        self.backbone_configs = backbone_configs
        self.components_name = components_name
        self.components_configs = components_configs

        self.backbone = get_backbone_components.get_backbone(backbone_arch=self.backbone_name,backbone_configs=self.backbone_configs)
        self.component = get_backbone_components.get_components(compo_arch=self.components_name,compo_configs=self.components_configs)

        self.MSEloss = utils.loss_func.KD_MSE_loss()
        # self.MSEloss = nn.MSELoss()
        # self.L1loss = nn.L1Loss()
        self.KLloss = DistillKL(T=1.5)
        self.CEloss = nn.CrossEntropyLoss()
        self.InfoNCEloss = loss_func.InfoNCE_loss(self.CEloss)
        self.tau = nn.Parameter(torch.ones([1]) * np.log(1 / 0.07))

        self.views = views
        self.optimizer = optimizer
        self.grad_optimizer_name = grad_optimizer_name
        self.lr = lr
        self.lr_sched = lr_sched
        self.lr_sched_config = lr_sched_config
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        # self.alignment_unit = get_backbone_components.get_components(compo_arch='alignment_unit',compo_configs={'input_dim':512,
        #                                                                                                         'hidden_dim':256,
        #                                                                                                         'output_dim':512})
        self.save_hyperparameters()

    def forward(self,x):
        x = self.backbone(x)
        x = self.component(x)
        return x

    def configure_optimizers(self):

        scaler = torch.cuda.amp.GradScaler()

        if self.optimizer.lower() == 'sgd':
            base_optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                             weight_decay=self.weight_decay,
                                             momentum=self.momentum,
                                             nesterov=True)
        elif self.optimizer.lower() == 'adamw':
            base_optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer.lower() == 'adam':
            base_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        if self.grad_optimizer_name not in [None, False]:
            if 'pcgrad' in self.grad_optimizer_name.lower():
                self.grad_optimizer = ugo.PCGradAMP(num_tasks=3, optimizer=base_optimizer, scaler=scaler,
                                                    reduction='sum', cpu_offload=False)
                optimizer = self.grad_optimizer.optimizer
            elif 'gradvac' in self.grad_optimizer_name.lower():
                self.grad_optimizer = ugo.GradVacAMP(num_tasks=3, optimizer=base_optimizer, scaler=scaler,
                                                     beta=1e-2, reduction='sum', cpu_offload=False)
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
                pg['lr'] = lr_scale * self.lr

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

    def get_all_T_output(self,inputs_sat,inputs_street,inputs_drone):
        t_out_sat = self.T_model(inputs_sat)
        t_out_street = None
        t_out_drone = self.T_model(inputs_drone)
        if self.views == 3:
            t_out_street = self.T_model(inputs_street)
        return t_out_sat,t_out_street,t_out_drone

    def get_all_S_output(self,inputs_sat,inputs_street,inputs_drone):
        s_out_sat = self(inputs_sat)
        s_out_street = None
        s_out_drone = self(inputs_drone)
        if self.views == 3:
            s_out_street = self(inputs_street)
        return s_out_sat,s_out_street,s_out_drone

    # @staticmethod
    # def z_mean(desc):
    #     mean = desc.mean(dim=-2,keepdims=True)
    #     stdv = desc.mean(dim=-2,keepdims=True)
    #     return (desc-mean) / (1e-7 + stdv)

    def training_step(self, batch, batch_idx):
        '''
        Didn't calculate the street loss, if necessary, just extent it
        '''
        ((inputs_sat, labels_sat), (inputs_street, labels_street),
         (inputs_drone, labels_drone)) = batch
        self.labels_sat.append(labels_sat)
        self.labels_street.append(labels_street)
        self.labels_drone.append(labels_drone)
        t_out_sat, t_out_street, t_out_drone = self.get_all_T_output(inputs_sat,inputs_street,inputs_drone)
        s_out_sat, s_out_street, s_out_drone = self.get_all_S_output(inputs_sat,inputs_street,inputs_drone)

        t_logit_out_sat, t_logit_out_drone = t_out_sat[1], t_out_drone[1]
        s_logit_out_sat, s_logit_out_drone = s_out_sat[1], s_out_drone[1]

        # t_desc_out_sat, t_desc_out_drone = t_out_sat[3], t_out_drone[3]
        # s_desc_out_sat, s_desc_out_drone = s_out_sat[3], s_out_drone[3]

        t_feature_out_sat, t_feature_out_drone = torch.stack(t_out_sat[2],dim=2), torch.stack(t_out_drone[2],dim=2)
        s_feature_out_sat, s_feature_out_drone = torch.stack(s_out_sat[2],dim=2), torch.stack(s_out_drone[2],dim=2)


        # t_dout_sat = torch.stack(t_out_sat[1],dim=-1)
        # t_dout_drone = torch.stack(t_out_drone[1],dim=-1)

        cls_loss = (Loss.cal_loss(s_logit_out_sat, labels_sat, self.CEloss) +
                    Loss.cal_loss(s_logit_out_drone, labels_drone,self.CEloss))

        contrast_loss = self.InfoNCEloss(s_feature_out_sat, t_feature_out_sat, self.tau)
        contrast_loss += self.InfoNCEloss(s_feature_out_drone, t_feature_out_drone, self.tau)

        # sat_mseloss = cal_mse_loss(s_feature_out_sat, t_feature_out_sat, self.MSEloss)
        # drone_mseloss = cal_mse_loss(s_feature_out_drone, t_feature_out_drone, self.MSEloss)

        # sat_kl_loss = cal_kl_loss(s_logit_out_sat, t_logit_out_sat, self.KLloss)
        # drone_kl_loss = cal_kl_loss(s_logit_out_drone, t_logit_out_drone, self.KLloss)

        # loss =  0.1*cls_loss + 0.9*(sat_kl_loss + drone_kl_loss)
        loss = 0.1*cls_loss + 0.9*contrast_loss
        # loss = contrast_loss

        # 累积损失和准确率
        self.log('cls_loss', cls_loss.item(), prog_bar=True, logger=True)
        self.log('contrast_loss', contrast_loss.item(), prog_bar=True, logger=True)
        # self.log('sat_mse_loss', sat_mseloss.item(), prog_bar=True, logger=True)
        # self.log('drone_mse_loss', drone_mseloss.item(), prog_bar=True, logger=True)
        # self.log('sat_kl_loss', sat_kl_loss.item(), prog_bar=True, logger=True)
        # self.log('drone_kl_loss', drone_kl_loss.item(), prog_bar=True, logger=True)
        self.log('losses', loss.item(), prog_bar=True, logger=True)

        return {'loss': loss}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # we empty the batch_acc list for next epoch
        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        else:
            lr_schedulers.step()

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # 调用父类的 `state_dict` 方法
        original_state = super().state_dict()
        # 删除与 `T_model` 相关的权重
        keys_to_remove = [key for key in original_state if key.startswith("T_model.")]
        for key in keys_to_remove:
            del original_state[key]
        return original_state


def main():
    H = 224
    x = torch.randn(20, 3, H, H).to('cuda')

    # T_model_configs = load_config('/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/model_configs/dino_b_QDFL.yaml')['model_configs']
    model_configs = load_config('../../../../../../../media/whu/Filesystem2/codespace/KD_CVGL/model_configs/KD_dino_b_QDFL_resnet50_QDFL.yaml')
    # S_model_configs = load_config('/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/model_configs/resnet50_QDFL.yaml')['model_configs']

    m = Knowledge_Distill_Model(T_model_configs=model_configs['T_model_configs'],
                            T_model_weights_path=model_configs['T_model_weights_path'],
                            **model_configs['model_configs']).cuda()
    print(f"Input device: {x.device}")
    print(f"Teacher model device: {next(m.T_model.parameters()).device}")

    z = m(x)
    z1 = m.T_model(x)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    if isinstance(z, tuple):
        print(f'Student Training Output shape is cls:{torch.stack(z[0], dim=0).shape},feature:{torch.stack(z[1], dim=0).shape}')
    else:
        print(f'Student Training Output shape is {torch.stack(z, dim=0).shape}')

    if isinstance(z1, tuple):
        print(f'Teacher Training Output shape is cls:{torch.stack(z1[0], dim=0).shape},feature:{torch.stack(z1[1], dim=0).shape}')
    else:
        print(f'Teacher Training Output shape is {torch.stack(z1, dim=0).shape}')

    m.train(False)
    z = m(x)
    if isinstance(z, tuple):
        print(
            f'Evaluating Output shape is cls:{torch.stack(z[0], dim=0).shape},feature:{torch.stack(z[1], dim=0).shape}')
    else:
        print(f'Evaluating Output shape is {z.shape}')

if __name__ == "__main__":
    main()
