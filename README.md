# dinov2-finetune
Finetuning DINOv2 (https://github.com/facebookresearch/dinov2) on your customized dataset.

First download dinov2: `git clone https://github.com/facebookresearch/dinov2.git`, 
and then run finetuning by 

```python dinov2_finetune.py --arch dinov2_vitb14 --data-dir PATH_TO_DATASET```. 

`--arch` can also be `dinov2_vitl14`.
