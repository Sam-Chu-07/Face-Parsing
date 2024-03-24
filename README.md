# Computer Vision Course Final Project - CelebAMask-HQ Face Parsing

### Dependencies
- Pytorch 1.7.1
- torchvision 0.8.2
- Python 3.8
- numpy
- Pillow
- tqdm
- opencv-python
- tensorboardX
- pandas
- inplace_abn (https://github.com/mapillary/inplace_abn.git)

### Preprocessing
1. Prepare Dataset: Download CelebAMask-HQ dataset (https://github.com/switchablenorms/CelebAMask-HQ).

2. Move the mask folder, image folder, and CelebA-HQ-to-CelebA-mapping.txt file from CelebAMask-HQ into the *Data_preprocessing* folder.

3. Preprocess the data using the command 
```python g_mask.py```. If multiprocess is supported, use the command ```python g_mask.py --num_process 4``` to utilize four processes.

```python g_partition.py``` to partition the data into train set, test set, and validation set.

4. Create an *unseen* folder under Data_preprocessing and download the necessary unseen data into this folder.

5. (Optional) To add the Face Synthetics dataset to the existing train set:

5-1. Download the required Face Synthetics data from https://github.com/microsoft/FaceSynthetics?tab=readme-ov-file.

5-2. Extract the downloaded files into Data_preprocessing and rename the extracted folder to *Synthetic*.

5-3. Use the command python synthetic_preprocess.py to add the Face Synthetics data to the train set.

### Training
```
python -u main.py --batch_size 16 --imsize 512 --train --arch FaceParseNet50
```

### Pretrained Model
Initial weights file for EHANet: *https://drive.google.com/drive/folders/1VIcmb4qF7sbyLSEouNRaxKLpPJRhUIGW?usp=drive_link*

Weights file for EHANet with CutPaste Augmentation:  *https://drive.google.com/drive/folders/1yU-7fQJsws2IzN64K02U2yDiyySHS1Gy?usp=drive_link*

Weights file for EHANet with CutPaste Augmentation and FaceSynthetics Data: *https://drive.google.com/drive/folders/1pW1kVny5pq3EbbgfcINAW66NddETfyl6?usp=drive_link*

### Testing
Test on testing data
```
python -u main.py --arch FaceParseNet50 --imsize 512 --model_path ./models/FaceParseNet50_aug_sys_size512/best.pth
```


Test on unseen data
```
python -u main.py --arch FaceParseNet50 --imsize 512 --unseen --test_image_path ./Data_preprocessing/unseen --model_path ./models/FaceParseNet50_aug_sys_size512/best.pth
```


Test on unseen data with cropped images (width 450, height 512)
```
python -u main.py --arch FaceParseNet50 --imsize 512 --unseen --test_image_path ./Data_preprocessing/unseen --model_path ./models/FaceParseNet50_aug_sys_size512/best.pth --crop_w 450 --crop_h 512
```

The final results will be recorded in mask.csv.

## References
https://github.com/TracelessLe/FaceParsing.PyTorch





