# CoTrFuse

## 0. Copy this folder
- I did a copy with: 
1. code
2. open github desktop
3. selected a folder

## 1.Download pre-trained swin transformer model (Swin-T)
- Model already downloaded, it is in the 'pretrained_ckpt' directory. Also the 'checkpoint' directory has been created

## 2. Datasets 
- Datasets are in the 'datasets' folder (ISIC 2017 and COVID, but the last one has no csv)

## 3. Environment

   * Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

   I used Python 3.8 but I guess conda could work

## 4. Run the code

    ```python ISIC2017_segmentation_train.py --model_name "modelname"``` while inside the CoTrFuse folder.