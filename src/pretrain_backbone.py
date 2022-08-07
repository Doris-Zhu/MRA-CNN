import argparse
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision

from data.data_loader import CUB200_loader
from utils.logger import Logger
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def test_model(net, dataloader):
    log.log('Testing...')

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    test_length = 0
    for step, (inputs, labels) in enumerate(dataloader):
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        with torch.no_grad():
            pred = net(inputs)
            test_length += len(inputs)
            correct_top1 += torch.eq(pred.topk(1, 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top3 += torch.eq(pred.topk(3, 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top5 += torch.eq(pred.topk(5, 1, True, True)[1], labels.view(-1, 1)).sum().float().item()

    log.log(f'\tAccuracy@top1 = {correct_top1/test_length:.5%}')
    log.log(f'\tAccuracy@top3 = {correct_top3/test_length:.5%}')
    log.log(f'\tAccuracy@top5 = {correct_top5/test_length:.5%}')


def pretrain_backbone():
    log.log('Start of the program...')

    # configure backbone
    if args.backbone in ['mobilenet', 'mobilenetv2', 'mobilenet_v2']:
        backbone = torchvision.models.mobilenet.mobilenet_v2
    elif args.backbone in ['efficientnet', 'efficientnetv2', 'efficientnetv2s', 'efficientnet_v2_s']:
        backbone = torchvision.models.efficientnet.efficientnet_v2_s
    else:
        return

    state_dict = backbone(pretrained=True).state_dict()
    net = backbone(num_classes=200).to(device)
    state_dict['classifier.1.weight'] = net.state_dict()['classifier.1.weight']
    state_dict['classifier.1.bias'] = net.state_dict()['classifier.1.bias']
    net.load_state_dict(state_dict)

    cudnn.benchmark = True

    # load CUB data
    log.log('Loading CUB data')
    train_data = CUB200_loader(args.data_path, split='train')
    test_data = CUB200_loader(args.data_path, split='test')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, collate_fn=train_data.CUB_collate)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, collate_fn=test_data.CUB_collate)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # pretrain backbone
    log.log(f'Start training for {args.epoch} epochs')
    for epoch in range(args.epoch):
        losses = 0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss
            if step % 20 == 19:
                avg_loss = losses / 20
                log.log(f'loss @ step[{step:04d}/{len(train_loader)}]-epoch{epoch}: {loss:.10f}\t' +
                        f'avg loss (last 20): {avg_loss:.10f}')
                losses = 0

        # test every 10 epochs
        if epoch % 10 == 9:
            test_model(net, test_loader)

        if epoch % args.checkpoint == args.checkpoint - 1:
            stamp = f'e{epoch + 1}{int(time.time())}'
            torch.save(net, os.path.join(args.save_root, f'{args.backbone}-cub200-{stamp}.pt'))
            torch.save(optimizer.state_dict, os.path.join(args.save_root, f'optimizer-{stamp}.pt'))

    log.log('End of the program, exiting...')


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backbone',
        default='efficientnet',
        type=str,
        help='Name of the backbone to be used in RA-CNN'
    )
    parser.add_argument(
        '--checkpoint',
        default=20,
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
        default=80,
        type=int,
        help='Number of epochs'
    )
    parser.add_argument(
        '--save-root',
        default='build',
        type=str,
        help='Path where all weights and checkpoints are saved'
    )

    args = parser.parse_args()

    # check arguments
    supported_backbones = [
        'mobilenet', 'mobilenetv2', 'mobilenet_v2',
        'efficientnet', 'efficientnetv2', 'efficientnetv2s', 'efficientnet_v2_s',
    ]
    if args.backbone not in supported_backbones:
        print('This backbone is not yet supported!')
        sys.exit(0)

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
    log = Logger(os.path.join('log', f'backbone_{int(time.time())}.log'))

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # run pretrain
    pretrain_backbone()
