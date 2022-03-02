import timm
import argparse
from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='efficientnet_b5')
    args = parser.parse_args()

    m = timm.create_model(args.name, pretrained=False)
    summary(m.cuda(), (3, 512, 384))

