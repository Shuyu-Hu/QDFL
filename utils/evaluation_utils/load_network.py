from plModules.U1652_baseline import U1652_model
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_network_supervised(configs, checkpoint):
    save_filename = checkpoint
    model = U1652_model(**configs)
    state_dict = torch.load(save_filename)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)