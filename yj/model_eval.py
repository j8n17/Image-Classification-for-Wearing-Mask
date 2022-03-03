import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import EvalDataset, MaskLabelDataset, ModelEvalDataset


def load_model(model_dir, num_classes, model_name, device):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(model_dir, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def eval(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskLabelDataset.num_classes  # 18

    if args.test_model == 'mask_label':
        model = load_model(model_dir, 3, args.model, device).to(device)
        dataset = EvalDataset(data_dir=args.data_dir, ratio=args.eval_ratio)
    else:
        model = load_model(model_dir, 6, args.model, device).to(device)
        dataset = ModelEvalDataset(data_dir=args.data_dir, ratio=args.eval_ratio, label=args.test_model)
    
    model.eval()
    
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
        for images, label in loader:
            images = images.to(device)
            if args.test_model == 'mask_label':
                labels.append(label[0] // 6)
            else:
                labels.append(label[0] - label[0] // 6)
            
            pred = model(images)
            pred = pred.argmax(dim=-1)
            # print(pred)

            # preds.extend(pred.cpu().numpy())
            preds.append(pred)

    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    acc = (labels == preds).sum().item() / len(loader)
    # info['ans'] = preds
    # info.to_csv(os.path.join(output_dir, f'output_googlenet.csv'), index=False)
    print(f'Evaluation Done! accuracy : {acc:4.2%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for validing (default: 1000)') ########
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='GoogLeNet', help='model type (default: BaseModel)')
    parser.add_argument('--augmentation', type=str, default='CropAugmentation', help='data augmentation type (default: BaseAugmentation)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default='./model/exp_incorrect_googlenet')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    ################################# my args
    parser.add_argument('--test_model', type=str, default='incorrect_mask')
    parser.add_argument('--eval_ratio', type=float, default=0.5)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    eval(data_dir, model_dir, output_dir, args)