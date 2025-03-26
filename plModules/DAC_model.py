from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
import utils
import utils.cal_loss as Loss
from plModules.U1652_baseline import U1652_model
import utils.grad_optim as ugo
from torch.optim import lr_scheduler, Optimizer
from transformers import get_cosine_schedule_with_warmup

class DAC_model(U1652_model):
    def __init__(self, backbone_name,
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
                 warmpup_steps=0):
        super().__init__(backbone_name,
                 backbone_configs,
                 components_name,
                 components_configs,
                 views,
                 optimizer,
                 grad_optimizer_name,
                 loss_config,
                 lr,
                 lr_sched,
                 lr_sched_config,
                 weight_decay,
                 momentum,
                 warmpup_steps)
        
        if self.add_MSEloss:
            self.MSEloss = utils.loss_func.DAC_MSE_loss()

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
        #[dsa_feature,cls_feature,contrast_feature,t,x]
        if contrast:
            if 'triplet' in self.which_contrast_loss.lower():
                contrast_loss = Loss.cal_triplet_loss(out_sat[2], out_drone[2], labels_sat, self.Tripletloss)
                contrast_loss += Loss.cal_triplet_loss(out_drone[2], out_sat[2], labels_sat, self.Tripletloss)
            elif 'multi' in self.which_contrast_loss.lower():
                contrast_loss = 2 * Loss.cal_MS_loss(out_sat[2], out_drone[2], labels_sat, self.MSloss, self.MSminer)
                # contrast_loss += Loss.cal_MS_loss(out_drone[1], out_sat[1], labels_sat, self.MSloss, self.MSminer)
            elif 'infonce' in self.which_contrast_loss.lower():
                contrast_loss = self.InfoNCEloss(out_sat[3], out_drone[3], self.tau)
                # contrast_loss += self.InfoNCEloss(out_drone[2], out_sat[2], self.tau)
        else:
            contrast_loss = 0

        if self.add_MSEloss:
            # In order to realise domain generalization, extract the fusion_feature to calculate mmd loss
            mseloss = Loss.cal_mse_loss(out_sat[0], out_drone[0], self.MSEloss)
        else:
            mseloss = torch.tensor((0))

        # out_sat/out_drone[0] indicates the output cls digit
        out_sat, out_drone = out_sat[1], out_drone[1]
        if not isinstance(out_sat, list):
            out_sat, out_drone = [out_sat[1]], [out_drone[1]]
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
                klloss = 0.5 * Loss.cal_js_loss(out_drone, out_sat, self.KLloss)
                klloss += 0.5 * Loss.cal_js_loss(out_sat, out_drone, self.KLloss)
            else:
                klloss = torch.tensor((0))

        # 累积损失和准确率
        self.log('cls_loss', 0.1*cls_loss.item(), prog_bar=True, logger=True)
        if contrast:
            self.log(f'{self.which_contrast_loss}', contrast_loss.item(), prog_bar=True, logger=True)
        if self.add_KLloss:
            self.log('kl_loss', klloss.item(), prog_bar=True, logger=True)
        if self.add_MSEloss:
            self.log('MSEloss', 0.6*mseloss.item(), prog_bar=True, logger=True)

        if self.grad_optimizer_name:
            loss = [cls_loss, contrast_loss, klloss]
            self.grad_optimizer.zero_grad()
            self.grad_optimizer.backward(loss)

            self.grad_optimizer.step()
            self.log('avg_loss', sum(loss), prog_bar=True, logger=True)
            return {'loss': sum(loss)}
        else:
            loss = 0.1*cls_loss + contrast_loss + 0.6*mseloss
            self.log('avg_loss', loss.item(), prog_bar=True, logger=True)
            return {'loss': loss}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # we empty the batch_acc list for next epoch
        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        else:
            lr_schedulers.step()

    def on_train_epoch_end(self):
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