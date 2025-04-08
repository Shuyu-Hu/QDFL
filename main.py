import time
import tkinter as tk
from tkinter import messagebox
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from utils.Evaluation_Callback import EvaluateAllOnTrainEnd
import pytorch_lightning as pl
from plModules import *
from datasets.train import *
from utils.commons import load_config
from utils.MyTQDMProgressBar import MyTQDMProgressBar

torch.autograd.set_detect_anomaly(True)

progress_bar = MyTQDMProgressBar()
import imgaug as ia
pl.seed_everything(seed=1, workers=True)
#===============================================load config===============================================
options = load_config('./model_configs/dino_b_QDFL.yaml')
'''
换数据集训练的时候别忘了改num_classes
'''
# dm = SUES_200_DataModule(
#         batch_size=options['batch_size'],
#         image_size=options['image_size'],
#         sample_num=options['sample_num'],
#         height=250,
#         num_workers=4,
#         DAC_sampling=options['DAC_sampling'],
#         show_data_stats=True,
#         )

# dm = DenseUAVDataModule(
#         batch_size=options['batch_size'],
#         image_size=options['image_size'],
#         sample_num=1,
#         num_workers=4,
#         show_data_stats=True,
#         sources = ['satellite','drone'],
#         data_augmentation = {
#                     'rotate_crop':["uav"],
#                     'random_affine':["satellite"],
#                     'color_jittering':[None],
#                     'random_erasing':[None],
#                     'random_erasing_prob':0.5}
#         )

dm = U1652DataModule(
        batch_size=options['batch_size'],
        image_size=options['image_size'],
        sample_num=options['sample_num'],
        num_workers=4,
        DAC_sampling=options['DAC_sampling'],
        drop_last=options['drop_last'] if 'drop_last' in options else True,
        show_data_stats=True,
        sources = ['satellite', 'street', 'drone'],
        )

m = U1652_model(**options['model_configs'])
# m = DAC_model(**options['model_configs'])
# m = DenseUAV_model(**options['model_configs'])

checkpoint_cb = ModelCheckpoint(
        monitor='drone_train_acc',
        mode='max',
        filename='epoch({epoch:02d})_sat_acc({sat_train_acc:02f})_drone_acc({drone_train_acc:02f})',
        # filename='epoch({epoch:02d})_loss({cls_loss:02f})',
        save_top_k=2,
        save_on_train_epoch_end=True,
        save_weights_only=False,
        save_last=True,
        verbose=False
        )

evaluate_cb = EvaluateAllOnTrainEnd(base_configs=options['model_configs'],
                                    fliplr=True,
                                    img_size=options['test_size'],
                                    batch_size=192,
                                    datasets_configs={'U1652': ['sat->drone', 'drone->sat'],
                                                    # 'DenseUAV':['drone->sat'],
                                    # 'SUES200': ([150, 200, 250, 300], ['sat->drone', 'drone->sat'])
                                                      },
                                    # datasets_configs={'SUES200': ([250], ['sat->drone', 'drone->sat'])}
                                    )

trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        # default_root_dir=f'./LOGS/{model_dino.encoder_arch}/{model_dino.agg_arch}', # Tensorflow can be used to viz
        default_root_dir=f'./LOGS/{m.backbone_name}_{m.components_name}',  # test
        enable_checkpointing=True,
        precision='16-mixed',  # we use half precision to reduce memory usage
        max_epochs=options['max_epochs'],
        # check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[progress_bar, checkpoint_cb, evaluate_cb],
        # callbacks=[checkpoint_cb, progress_bar],
        deterministic='warn',
        # callbacks=[checkpoint_cb,early_stopping],# if you want to use early stopping, just uncomment it and comment the above line
        log_every_n_steps=10,
        fast_dev_run=False  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
        # limit_train_batches=100
        )

# 计时开始
start_time = time.time()

# 模型训练部分
if dm is None:
    print("trainer.lightning_module is None!")
    # 可能需要在这里添加更多的调试信息或错误处理
else:
    '''
    if you want to resume training process, please use the first sentence
    '''
    # trainer.fit(model=m, datamodule=dm,
    #             ckpt_path='/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/LOGS/resnet50_DF/lightning_logs/version_0/checkpoints/last.ckpt')
    trainer.fit(model=m, datamodule=dm)

# 计算训练时间
end_time = time.time()
training_time = end_time - start_time

# 将训练时间转换为h:m:s格式
hours, rem = divmod(training_time, 3600)
minutes, seconds = divmod(rem, 60)
formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

# 显示训练完成通知及训练时间
def show_notification():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showinfo("Complete", f"Total time: {formatted_time}")

show_notification()
