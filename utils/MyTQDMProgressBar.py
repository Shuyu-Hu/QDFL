import sys
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=0,  # 这里固定写0
            disable=self.is_disabled,
            leave=True,  # leave写True
            dynamic_ncols=True,
            file=sys.stdout,
        )
        bar.set_description('running validation ...')
        return bar
