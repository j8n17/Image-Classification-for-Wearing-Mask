# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
`SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images SM_MODEL_DIR=/opt/ml/exps/ python train.py --name`
`python train.py --epochs 30 --model ShuffleNet --name shuffle_fullsize_epoch180_cutmix0.7_reg --beta 1 --cutmix_prob 0.7`
`python train.py --epochs 30 --model GoogLeNet --name google_fullsize_epoch180_cutmix0.7_reg --beta 1 --cutmix_prob 0.7`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
`python inference.py `
### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
