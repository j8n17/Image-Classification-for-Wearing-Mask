import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import *

from dataset import EvalDataset, MaskLabelDataset


def load_model(model_dir, saved_model, num_classes, model_name, device):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(model_dir, saved_model)
    model_path = os.path.join(model_path, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def eval(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskLabelDataset.num_classes  # 18
    
    mask_label_model = load_model(model_dir, args.mask_label_dir, 3, args.mask_label_model, device).to(device)
    mask_model = load_model(model_dir, args.mask_dir, 6, args.mask_model, device).to(device)
    normal_model = load_model(model_dir, args.normal_dir, 6, args.normal_model, device).to(device)
    incorrect_model = load_model(model_dir, args.incorrect_dir, 6, args.incorrect_model, device).to(device)
    
    mask_label_model.eval()
    mask_model.eval()
    normal_model.eval()
    incorrect_model.eval()

    dataset = EvalDataset(data_dir=args.data_dir, ratio=args.eval_ratio)
    transform_cls = getattr(import_module("dataset"), args.augmentation)
    # BaseAug
    transform = transform_cls(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating Evaluation results..")
    preds = []
    labels = []
    with torch.no_grad():
        # TTA_transform = transforms.Compose([
        #     ToPILImage(),
        #     CenterCrop((384, 384)),
        #     ToTensor(),
        # ])
        for images, label in loader:
            # TTA_image = TTA_transform(images[0])
            # TTA_image = TTA_image.unsqueeze(dim=0)
            images = images.to(device)
            # TTA_image = TTA_image.to(device)
            labels.append(label[0])
            
            # 마스크 착용 여부 예측
            mask_label_pred = mask_label_model(images)
            mask_label_pred = mask_label_pred.argmax(dim=-1)
            # print(pred)

            # 마스크 착용 여부에 따라 다른 모델 적용
            if mask_label_pred[0] == 0:
                gender_age_pred = mask_model(images)
                # TTA_pred = mask_model(TTA_image)
            elif mask_label_pred[0] == 1:
                gender_age_pred = incorrect_model(images)
                # TTA_pred = incorrect_model(TTA_image)
            else:
                gender_age_pred = normal_model(images)
                # TTA_pred = normal_model(TTA_image)
            
            # gender_age_pred = gender_age_pred + TTA_pred
            gender_age_pred = gender_age_pred.argmax(dim=-1)

            pred = mask_label_pred * 6 + gender_age_pred
            # preds.extend(pred.cpu().numpy())
            preds.append(pred)

    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    acc = (labels == preds).sum().item() / len(loader)
    # info['ans'] = preds
    # info.to_csv(os.path.join(output_dir, f'output_googlenet.csv'), index=False)
    print(f'Evaluation Done! accuracy : {acc:4.2%}')

    for i in range(len(labels)):
        if labels[i] != preds[i]:
            print(f'label : {labels[i]}, pred : {preds[i]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for validing (default: 1000)') ########
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='GoogLeNet', help='model type (default: BaseModel)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    ################################# my args
    parser.add_argument('--mask_label_dir', type=str, default='exp_mask_label') # 마스크 착용 여부 모델 dir
    parser.add_argument('--mask_dir', type=str, default='exp_mask_googlenet') # 마스크를 쓴 경우 성별, 나이 모델 dir
    parser.add_argument('--normal_dir', type=str, default='exp_normal_googlenet') # 마스크를 쓴 경우 성별, 나이 모델 dir
    parser.add_argument('--incorrect_dir', type=str, default='exp_incorrect_googlenet') # 마스크를 쓴 경우 성별, 나이 모델 dir

    parser.add_argument('--mask_label_model', type=str, default='ResNet18')
    parser.add_argument('--mask_model', type=str, default='GoogLeNet')
    parser.add_argument('--normal_model', type=str, default='GoogLeNet')
    parser.add_argument('--incorrect_model', type=str, default='GoogLeNet')

    parser.add_argument('--eval_ratio', type=float, default=0.5)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    eval(data_dir, model_dir, output_dir, args)