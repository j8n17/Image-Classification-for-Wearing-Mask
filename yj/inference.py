import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskLabelDataset


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
def inference(data_dir, model_dir, output_dir, args):
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

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            
            # 마스크 착용 여부 예측
            mask_label_pred = mask_label_model(images)
            mask_label_pred = mask_label_pred.argmax(dim=-1)
            # print(pred)

            # 마스크 착용 여부에 따라 다른 모델 적용
            if mask_label_pred[0] == 0:
                gender_age_pred = mask_model(images)
                gender_age_pred = gender_age_pred.argmax(dim=-1)
            elif mask_label_pred[0] == 1:
                gender_age_pred = incorrect_model(images)
                gender_age_pred = gender_age_pred.argmax(dim=-1)
            else:
                gender_age_pred = normal_model(images)
                gender_age_pred = gender_age_pred.argmax(dim=-1)

            pred = mask_label_pred * 6 + gender_age_pred
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output_googlenet.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for validing (default: 1000)') ########
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='GoogLeNet', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
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

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
