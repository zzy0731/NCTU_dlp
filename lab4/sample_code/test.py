from matplotlib.style import available
import torch
from dataset import bair_robot_pushing_dataset
from torch.utils.data import DataLoader
from train_fixed_prior import parse_args
from utils import pred,finn_eval_seq,plot_pred
import numpy as np
from models.vgg_64 import vgg_decoder, vgg_encoder
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
args=parse_args()
torch.manual_seed(args.seed)

test_data = bair_robot_pushing_dataset(args, 'test')

test_loader = DataLoader(test_data,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True)
test_iterator = iter(test_loader)

modules =torch.load('logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/model.pth')




# psnr_list = []
# for i in range(len(test_data) // args.batch_size):
#     try:
#         test_seq, test_cond = next(test_iterator)
#     except StopIteration:
#         test_iterator = iter(test_loader)
#         test_seq, test_cond = next(test_iterator)
    
#     pred_seq = pred(test_seq, test_cond, modules, args, device)
#     test_seq = test_seq.permute(1,0,2,3,4)
#     _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
#     psnr_list.append(psnr)
#     print(i)
    
# ave_psnr = np.mean(np.concatenate(psnr_list))
# print('test_psnr',ave_psnr)

name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

args.log_dir = '%s/%s' % (args.log_dir, name)
try:
    test_seq, test_cond = next(test_iterator)
except StopIteration:
    test_iterator = iter(test_loader)
    test_seq, test_cond = next(test_iterator)
test_seq = test_seq.to(device)
test_cond = test_cond.to(device)
epoch=350
psnr=plot_pred(test_seq, test_cond, modules, epoch, args,lasttext='test')
print(psnr.shape)


ave_psnr = np.mean(np.concatenate(psnr))
print(ave_psnr)

