from shutil import which

import torch
from pytorch_lightning.callbacks import Callback
from Supervised_evaluate import evaluate_all_supervised_learning
from datetime import datetime

WHICH_MODE = {
    'metric_learning',
    'supervised'
}

class EvaluateAllOnTrainEnd(Callback):
    def __init__(self, base_configs, fliplr, img_size, batch_size, datasets_configs=None, which_mode='supervised'):
        super().__init__()
        assert which_mode in WHICH_MODE
        self.which_mode = which_mode
        if datasets_configs in [None,False]:
            self.datasets_configs = {'U1652': ['sat->drone', 'drone->sat'],
                                    'SUES200': ([150, 200, 250, 300], ['sat->drone', 'drone->sat'])}
        else:
            self.datasets_configs = datasets_configs
        self.base_configs = base_configs
        self.fliplr = fliplr
        self.img_size = img_size
        self.batch_size = batch_size

    def on_train_end(self, trainer, pl_module):
        if pl_module.grad_optimizer_name:
            # 获取训练后的模型权重路径
            timestamp = datetime.now().strftime("%H%M%S")
            pth_path = f"{trainer.logger.save_dir}/lightning_logs/{pl_module.grad_optimizer_name}/checkpoints/epoch={pl_module.current_epoch}_time={timestamp}.ckpt"

            pl_module.trainer.save_checkpoint(pth_path)
        else:
            pth_path = trainer.checkpoint_callback.last_model_path

        torch.cuda.empty_cache()

        if not pth_path:
            print("No checkpoint was saved, skipping evaluation.")
            return

        # 在训练结束时评估模型
        evaluate_all_supervised_learning(
            datasets_configs=self.datasets_configs,
            base_configs=self.base_configs,
            fliplr=self.fliplr,
            img_size=self.img_size,
            batch_size=self.batch_size,
            pth_path=pth_path
        )