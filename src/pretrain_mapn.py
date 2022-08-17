import argparse
import imageio
import matplotlib.pyplot as plt
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from data.data_loader import CUB200_loader
from utils.logger import Logger
from model import RACNN
from torch.utils.data import DataLoader


def save_img(x, path, annotation=''):
    fig = plt.gcf()  # generate outputs
    plt.imshow(CUB200_loader.tensor_to_img(x[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image


def build_gif(pattern='@2x', sample=0, gif_name='pretrain_apn_cub200', cache_path='build/img'):
    log.log(f'Saving gif pattern {pattern} sample {sample}')
    files = [x for x in os.listdir(cache_path) if f"_sample_{sample}{pattern}" in x]
    files.sort(key=lambda x: int(x.split('@')[0].split('_')[1]))
    gif_images = [imageio.imread(f'{cache_path}/{img_file}') for img_file in files]
    imageio.mimsave(f"build/{gif_name}{pattern}_{str(sample)}-{int(time.time())}.gif", gif_images, fps=8)


def pretrain_apn():
    net = RACNN(num_classes=200).to(device)
    if args.pretrained_backbone:
        state_dict = torch.load(args.pretrained_backbone).state_dict()
        net.convolution1.load_state_dict(state_dict)
        net.convolution2.load_state_dict(state_dict)
        net.convolution3.load_state_dict(state_dict)
    cudnn.benchmark = True

    params = list(net.mapn.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    # load CUB data
    log.log('Loading CUB data')
    train_data = CUB200_loader(args.data_path, split='train')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_data.CUB_collate)

    test_data = CUB200_loader(args.data_path, split='test')
    test_loader = DataLoader(
        test_data, batch_size=4, shuffle=True, num_workers=4, collate_fn=test_data.CUB_collate)
    
    load, _ = next(iter(test_loader))

    net.mode("pretrain_apn")

    for epoch in range(args.epoch):
        loss_sum = []
        for step, (inputs, _) in enumerate(train_loader):
            loss = net.echo(inputs, optimizer)
            loss_sum.append(loss)
            avg_loss = sum(loss_sum[-5 if len(loss_sum) > 5 else -len(loss_sum):]) / len(loss_sum[-5 if len(loss_sum) > 5 else -len(loss_sum):])
            log.log(f'loss @ step [{step:4d}]: {loss:.4f}, avg loss : {avg_loss:.4f}')

            # sample_idx = 0
            # for sample in load:
            #     _, _, _, resized = net(sample.unsqueeze(0).cuda())
            #     x1, x2 = resized[0].data, resized[1].data
            #     save_img(x1, path=f'build/img/step_{step}_sample_{sample_idx}@2x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {step}')
            #     save_img(x2, path=f'build/img/step_{step}_sample_{sample_idx}@4x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {step}')
            #     sample_idx += 1

            if step * args.batch_size > 3200:
                log.log(f'stopping @ step [{step:4d}]')
                break

        if epoch % args.checkpoint == args.checkpoint - 1:
            torch.save(net.state_dict(), f'build/mapn-{int(time.time())}.pt')
        #     for i in range(4):
        #         build_gif(pattern='@2x', sample=i, gif_name='pretrain_apn_cub200')
        #         build_gif(pattern='@4x', sample=i, gif_name='pretrain_apn_cub200')


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained-backbone',
        type=str,
        help='Path of pretrained backbone'
    )
    parser.add_argument(
        '--batch-size',
        default=16,
        type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Save a checkpoint every # epochs'
    )
    parser.add_argument(
        '--data-path',
        default=os.path.join('external', 'CUB_200_2011'),
        type=str,
        help='Data root directory'
    )
    parser.add_argument(
        '--epoch',
        default=1,
        type=int,
        help='Number of epochs'
    )
    parser.add_argument(
        '--save-path',
        default='build',
        type=str,
        help='Path where all weights and checkpoints are saved'
    )

    args = parser.parse_args()

    if args.checkpoint <= 0 or args.epoch <= 0:
        print('Invalid checkpoint or epoch argument')
        sys.exit(0)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print('Suitable cuda device not found!')
        sys.exit(0)

    # configurate logger
    if not os.path.exists(os.path.join('log')):
        os.makedirs(os.path.join('log'))
    log = Logger(os.path.join('log', f'apn_{int(time.time())}.log'))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # run pretrain
    pretrain_apn()

    # TODO: gif visualization
