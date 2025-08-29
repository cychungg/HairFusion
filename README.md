# HairFusion (AAAI2025)

This repository is the official implementation of [What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer](https://arxiv.org/abs/2408.16450).
Note that this repository only includes the inference codes.

![alt text](asset/figure-teaser.png)

![alt text](asset/figure-grid.png)


## Environments

Install PyTorch and other dependencies:

```
conda create -y -n hairfusion python=3.8
conda activate hairfusion
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install einops
pip install pytorch-lightning==1.5.0
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install omegaconf
pip install albumentations
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install clean-fid==0.1.35
pip install diffusers==0.20.2
pip install scipy==1.10.1
conda install -c anaconda ipython -y
```


## Sample Dataset

sample data in ./data/test

## Custom Data
### 1) Preprocessing
Preprocess a given image and save the outputs in ./data/${dir_name}
- Download [face_segment16.pth (50.8MB)](https://drive.google.com/file/d/10GL030sNpVrxM9Ez0nXhHvs9-lsnZFGV/view?usp=sharing) and [shape_predictor_68_face_landmarks.dat (95.1MB)](https://drive.google.com/file/d/1g4jTab8cNVmF2AjDz2N3uXu0cMvsvlC3/view?usp=sharing) into ./models
- run 
  ```bash
  CUDA_VISIBLE_DEVICES=0 python preprocess.py --img_path ${image_path} --save_dir_name ${dir_name} --crop_scale ${crop_scale}
  ```

### 2) DensePose Extraction
Extract DensePose of the images in ./data/${dir_name}/images and save them in ./data/${dir_name}/images-densepose
- Installation
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy
    cd detectron2
    pip install -e .
    cd projects/DensePose
    pip install -e .
    ```
- run in ./detectron2/project/DensePose:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python apply_net.py get configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl ../../../data/${dir_name}/images --output ../../../data/${dir_name}/images-densepose  -v
    ```
    Please check the ['detectron2' repository](https://github.com/facebookresearch/detectron2.git) for details.

### 3) Make Agnostic Images
- run:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python make_agnostic.py --dir_name ${dir_name} --dil_size ${hair_dilation_size}
  ```

### 4) Make Test Pairs
Save a text file of test pairs as ./data/${dir_name}/test_pairs.txt
```bash
python make_test_pairs.py --dir_path ./data/${dir_name}
```

## Checkpoints
- VAE model:
  - download 'realisticVisionV51_v51VAE.ckpt' (3.97G) from [https://civitai.com/models/4201?modelVersionId=130072](https://civitai.com/models/4201?modelVersionId=130072) and save it into ./models
- HairFusion:
  - download and unzip ['hairfusion.zip' (8.4G)](https://drive.google.com/file/d/1QWB478Fc415CSqvDDe4ZczGTzh2FAXTO/view?usp=sharing) and save it into ./logs

## Inference
```
bash ./scripts/test.sh
```

## Citation
If you find our work useful for your research, please cite us:
```
@inproceedings{chung2025hairfusion,
  title={What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer},
  author={Chung, Chaeyeon and Park, Sunghyun and Kim, Jeongho and Choo, Jaegul},
  booktitle={The Association for the Advancement of Artificial Intelligence},
  year={2025}
}
```

## License
Licensed under the CC BY-NC-SA 4.0 license [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

