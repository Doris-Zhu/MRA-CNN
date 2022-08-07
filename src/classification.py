import argparse
import torch
import cv2

from model import RACNN
from torch.autograd import Variable
from torchvision import transforms


def run():
    std = 1. / 255.
    means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=means,
            std=[std]*3)
    ])

    img1 = cv2.imread(args.img_path)
    img1 = transform(img1)

    img2 = cv2.imread('docs/cn.jpg')
    img2 = transform(img2)

    img3 = cv2.imread('docs/lf.jpg')
    img3 = transform(img3)

    img4 = cv2.imread('docs/rwb.jpg')
    img4 = transform(img4)

    imgs = torch.stack((img1, img2, img3, img4))
    inputs = Variable(imgs).cuda()

    model = RACNN(200).to('cuda:0')
    model.load_state_dict(torch.load(args.weight))

    scores = dict()
    with torch.no_grad():
        outputs, _, _, _ = model(inputs)
        for idx, logits in enumerate(outputs):
            for i, c in enumerate(logits):
                pred = c.topk(5, 0, True, True)
                for j in range(5):
                    scores[pred[1][j].item()] = scores.get(pred[1][j].item(), 0) + 5 - j
                break

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(sorted_scores)
    print(index_to_name(sorted_scores[0][0]))


def index_to_name(num):
    classes = open("external/CUB_200_2011/classes.txt", "r")
    content = classes.read()
    classes.close()
    content = content.splitlines()
    for line in content:
        index = int(line.split(' ')[0])
        if index == num + 1:
            return line.split('.')[1].strip().replace('_', ' ')
    return 'Error'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RA-CNN Classification')
    parser.add_argument(
        '--img-path',
        default='img/bobolink.jpg',
        type=str,
        help='Image path'
    )
    parser.add_argument(
        '-w',
        '--weight',
        default='build/racnn_efficientnet_v2_cub200-e20se201658759121.pt',
        type=str,
        help='Weight path'
    )
    args = parser.parse_args()
    run()
