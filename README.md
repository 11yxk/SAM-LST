# SAM-LST

Pytorch implementation of paper ''AAA''.



# Data Preparation
We borrow the data process from [SAMed](https://github.com/hitachinsk/SAMed).
# Training & Testing

### Training
```
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW 
```
### Testing

```
python test.py --is_savenii --output_dir <Your output directory> --my_ckpt <ckpt path>
```

### Pretrained Models

- Pretrained Models are available at .


## Acknowledgements

This repo is based on [SAMed](https://github.com/hitachinsk/SAMed) and [TransUnet](https://github.com/Beckschen/TransUNet).

Thanks original authors for their work!
