import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn
import progressbar
mse_criterion = nn.MSELoss()

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pred(x, cond, modules, args, device):
    x = x.permute(1,0,2,3,4)
    cond = cond.permute(1,0,2)

    x = x.to(device)
    cond = cond.to(device)

    
    # xinputlist=[]
    # xinputlist.append(x[0])
    # _, skip = modules['encoder'](x[0])
    # for i in range(1, args.n_eval):
    #     torch.cuda.empty_cache()
    #     xinput=xinputlist[i-1]
    #     h_input,_=modules['encoder'](xinput)#輸入的input x生成h
    #     h_target,_=modules['encoder'](x[i]) #取樣解答生成的h
    #     z_t, mu, logvar = modules['posterior'](h_target) #從取樣解答生成的h找出z
    #     h_pred = modules['frame_predictor'](torch.cat([h_input, z_t, cond[i-1]], 1))
    #     x_pred = modules['decoder']([h_pred,skip])
    #     print(x_pred.shape)
        
    #     xinputlist.append(x_pred)
    
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past)]
    gen_seq=[]
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, args.n_eval):
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
            h = h.detach()
            
        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1)) 
            x_in = x[i]
            # print(x_in.shape)
            gen_seq.append(x_in)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
            h = modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            #print(x_in.shape)
            gen_seq.append(x_in)
    

    gen_seq = torch.stack(gen_seq)
    return gen_seq

def plot_pred(x, cond, modules, epoch, args,lasttext=None):

    x = x.permute(1,0,2,3,4)
    cond = cond.permute(1,0,2)

    # get approx posterior sample
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, args.n_eval):
        h = modules['encoder'](x_in)
        h_target = modules['encoder'](x[i])[0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= modules['posterior'](h_target) # take the mean
        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1)).detach()
            x_in = modules['decoder']([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  

    nsample = 3
    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))
    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, args.n_past + args.n_future):
            h = modules['encoder'](x_in)
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < args.n_past:
                h_target = modules['encoder'](x[i])[0].detach()
                _, z_t, _ = modules['posterior'](h_target)
            else:
                z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
            if i < args.n_past:
                modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                h = modules['frame_predictor'](torch.cat([h, z_t, cond[i-1]], 1)).detach()
                x_in = modules['decoder']([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        # print('====')
        # print(len(gt_seq))
        # print(gt_seq[0].shape)
        _, ssim[:, s, :], psnr[:, s, :] = eval_seq(gt_seq, gen_seq)

    progress.finish()
    clear_progressbar()

    ###### ssim ######
    for i in range(args.batch_size):
        gifs = [ [] for t in range(args.n_eval) ]
        text = [ [] for t in range(args.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(args.n_past + args.n_future):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/gen/epoch=%d_%d_%s.gif' % (args.log_dir, epoch, i,lasttext)
        
        save_gif_with_text(fname, gifs, text)
    return psnr

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        try:
            img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
            img = img.cpu()
            img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
            images.append((img*255).astype(np.uint8))
        except:
            pass
    # print(len(images))
    # print(images[0].shape)
    # print(images[0][50][200][0])
    imageio.mimsave(filename, images, duration=duration)

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot")  and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)    