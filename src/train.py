import argparse
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from data.data_loader import CUB200_loader
from utils.logger import Logger
from model import RACNN
from torch.autograd import Variable
from torch.utils.data import DataLoader


def train(model, loader, optim, epoch, mode):
    model.mode(mode)
    log.log(f'Switch to {mode}')
    losses = []

    for step, (inputs, labels) in enumerate(loader):
        loss = model.echo(inputs, labels, optim)
        losses.append(loss)

        if step % 50 == 49:
            log.log(f'loss @ step [{step + 1}/{len(loader)}] @ epoch {epoch}: '
                    f'{loss:.10f}\taverage loss 20: {sum(losses[-20:]) / 20:.10f}')

    return sum(losses) / len(losses)


def test(model, loader):
    log.log(f'Testing...')
    stats = {
        'scale0': {'top-1': 0, 'top-5': 0},
        'scale1': {'top-1': 0, 'top-5': 0},
        'scale2': {'top-1': 0, 'top-5': 0}}
    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            outputs, _, _, _ = model(inputs)
            for idx, logits in enumerate(outputs):
                stats[f'scale{idx}']['top-1'] += \
                    torch.eq(logits.topk(1, 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
                stats[f'scale{idx}']['top-5'] += \
                    torch.eq(logits.topk(5, 1, True, True)[1], labels.view(-1, 1)).sum().float().item()

            # stop early to save time
            if step > 200:
                for scale in stats.keys():
                    for topk in stats[scale].keys():
                        log.log(f'\tAccuracy {scale} @ {topk} [{step}/{len(loader)}] = '
                                f'{stats[scale][topk]/((step+1)*int(inputs.shape[0])):.5%}')
                return


def main():
    log.log('Program starts...')
    model = RACNN(num_classes=200).to(device)
    # model.load_state_dict(torch.load(pretrained_apn_path))

    cudnn.benchmark = True

    classification_parameters = \
        list(model.convolution1.parameters()) + \
        list(model.convolution2.parameters()) + \
        list(model.convolution3.parameters()) + \
        list(model.fc1.parameters()) + \
        list(model.fc2.parameters()) + \
        list(model.fc3.parameters())

    # TODO: change to mapn parameters
    mapn_parameters = list(model.mapn.parameters())

    # TODO: switch to Adam
    classification_optim = optim.SGD(classification_parameters, lr=0.001, momentum=0.9)
    mapn_optim = optim.SGD(mapn_parameters, lr=0.001, momentum=0.9)

    log.log('Loading CUB data')
    train_data = CUB200_loader(args.data_path, split='train')
    test_data = CUB200_loader(args.data_path, split='test')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, collate_fn=train_data.CUB_collate)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, collate_fn=test_data.CUB_collate)

    for epoch in range(args.epoch):
        classification_loss = train(model, train_loader, classification_optim, epoch, 'backbone')
        mapn_loss = train(model, train_loader, mapn_optim, epoch, 'mapn')
        test(model, test_loader)

        if epoch % args.checkpoint == args.checkpoint - 1:
            stamp = f'e{epoch + 1}{int(time.time())}'
            torch.save(model.state_dict(), os.path.join(args.save_root, f'racnn-effv2-cub200-e{epoch}-s{stamp}.pt'))
            torch.save(classification_optim.state_dict(), os.path.join(args.save_root, f'clsoptim-s{stamp}.pt'))
            torch.save(mapn_optim.state_dict(), os.path.join(args.save_root, f'mapnoptim-s{stamp}.pt'))

    log.log('End of program...')


if __name__ == '__main__':
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        default=10,
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
        default=50,
        type=int,
        help='Number of epochs'
    )
    parser.add_argument(
        '-w',
        '--pretrain-weight',
        type=str,
        help='Path of weight'
    )
    parser.add_argument(
        '--save-root',
        default='build',
        type=str,
        help='Path where all weights and checkpoints are saved'
    )

    args = parser.parse_args()
    pretrained_apn_path = args.pretrain_weight

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print('Suitable cuda device not found!')
        sys.exit(0)

    # configurate logger
    if not os.path.exists(os.path.join('log')):
        os.makedirs(os.path.join('log'))
    log = Logger(os.path.join('log', f'train_{int(time.time())}.log'))

    main()
