# CoTrFuse

## 1.Download pre-trained swin transformer model (Swin-T)
   * [Get pre-trained model in this link]
      (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/" and create dir 'chechpoint' (yes, like this, don't know why),'test_log' in the root path.

## 2. Datasets 
   * You can also go to https://challenge.isic-archive.com/data/#2017 to acquire the ISIC2017 dataset. Process the label from the csv file for training. Change the imgs_train_path, imgs_val_path, imgs_test_path in the train_class_after_segmentation to the path of the corresponding path.
   * You can also go to https://www.kaggle.com/datasets/cf77495622971312010dd5934ee91f07ccbcfdea8e2f7778977ea8485c1914df to acquire the COVID-QU-Ex dataset.

   Please note that for ISIC 2017 you need all the steps, I am still trying to understand how does the code divide

   also note that there is a problem with the csv, you can find the modified version of csv here (the problem was that in the code are called `image_id` and in the .csv are called `image_name`, also it removes the extension of the pic and the code won't find the path to the image - maybe only windows problem but however)

   drive to all the csv(s) with `image_name` instead of `image_id`: https://drive.google.com/drive/u/3/folders/1TlDK3TDqLNQTV35m93yhjsI9XAnzg3NG
   if you need to modify - adding the extensino to the image - you can use `modCSV` or `modifyCSV.py` here in the repo

   I could not find any csv for the covid dataset

## 3. Environment

   * Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

   I used Python 3.8 but I guess conda could work

## 4. Code modification

    I modified several parts in the `train_ISIC.py` file, this part is written because my setup was not able to work on training images:

    ```
    # Funzione per caricare e ridimensionare le immagini
    def load_and_resize_images(file_paths, target_size):
        print("dentro load_and_resize images")
        images = []

        for file_path in file_paths:
            #print("dentro il for")
            img = cv2.imread(file_path)[:, :, ::-1]  # Carica e converte in RGB
            img_resized = cv2.resize(img, target_size)  # Ridimensiona l'immagine
            images.append(img_resized)
        print("Done")
        return images

    # Esempio di utilizzo
    target_size = (128, 128)  # Specifica la dimensione desiderata
    imgs_train = load_and_resize_images(train_imgs, target_size)
    imgs_val = load_and_resize_images(val_imgs, target_size)
    masks_train=load_and_resize_images(train_masks,target_size)
    masks_val=load_and_resize_images(val_masks,target_size)
    ```

    the original part is available above (commented) and here:
    
    ```
    # #---- ORIGINAL IMPLEMENTATION
    imgs_train = [cv2.imread(i)[:, :, ::-1] for i in train_imgs]
    masks_train = [cv2.imread(i)[:, :, 0] for i in train_masks]
    imgs_val = [cv2.imread(i)[:, :, ::-1] for i in val_imgs]
    masks_val = [cv2.imread(i)[:, :, 0] for i in val_masks]
    ```

    Please note that in the `train_ISIC.py` you need to modify the path in the initial part (parser) otherwise it won't work.

## 5. Run the code

    ```python train_ISIC.py```