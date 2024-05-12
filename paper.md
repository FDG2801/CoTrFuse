# Introduction

Code: https://github.com/BinYCn/CoTrFuse

Various deep learning methods, particularly those based on convolutional neural networks (CNNs) like U-Net, have been widely used for medical image segmentation, improving the field's accuracy and reliability. However, CNN-based approaches have limitations in capturing global semantic information due to their restricted receptive fields.

To address this, attention-based method were introduced to computer vision domain, such as ViT and Swin Transformer since they showed promise in image recognition, and Swin-Unet extended this to medical image segmentation. Transformer, for the moment, tend to overlook local details.

To combine the strengths of CNNs and Transformers, hybrid architectures like TransUNet and DS-TransUNet emerged, incorporating CNNs for spatial features and Transformers for global information encoding. TransFuse proposed a BiFusion module to combine CNN and Transformer features.

The paper introduces CoTrFuse, which leverages Swin Transformer and EfficientNet as dual encoders. It introduces the Swin Transformer and CNN Fusion module (STCF) to fuse global and local semantic information effectively before the skip connections, enhancing segmentation performance. Experimental results on skin lesion and COVID-19 infection segmentation datasets demonstrate that CoTrFuse outperforms state-of-the-art segmentation methods, making it a promising approach for medical image segmentation.

# Figures

Fig. 1: CoTrFuse architecture
Fig. 2: EfficientNet block working
Fig. 3: Swin transformer block explaination
Fig. 4: Swin and CNN fusion module
Fig. 5"scSE module
Fig. 6: ISIC 2017 dataset results
Table 1: % results (such as table 2, 3)
Fig. 7: COVID-QU-Ex dataset results
Table 4: study on different models
Fig. 8: segmentation results with different modalities
Table 5 and 6: ablation studies with different methods
Fig. 9: feature map 

# Conclusion

The experiments on two diverse datasets, ISIC-2017 and COVID-QU-Ex, demonstrate that CoTrFuse surpasses several state-of-the-art segmentation methods. These results emphasize the effectiveness of our approach, particularly in situations where precise segmentation plays a critical role in accurate medical diagnosis and treatment planning.

# Read but skip/skim math and parts that do not make sense

## Related works

### CNN-Based Segmentation Networks
These networks are designed to accurately segment regions in images while preserving spatial information. Notable models include:

- FCN-based models: They remove the fully connected layer to retain spatial information.
- DeepLabV3: Utilizes a cascaded deep ASPP module to merge multiple contexts.
- U-Net: Known for detailed and generalized image segmentation.
- UNet 3+: Introduces a full-scale jump connection for improved segmentation accuracy.
- FD-UNet: Incorporates dense connectivity to prevent redundancy and enhance information flow.
- CE-Net: Introduces Dense Atrous Convolution and Residual Multi-kernel pooling modules for advanced feature capture.
- 3D-UNet and V-Net: CNN-based models applied to 3D medical image segmentation.
CNN-based research in image segmentation has made significant progress.

### ViT

Initially developed for NLP, ViT has shown outstanding accuracy in image recognition tasks. To address limitations related to large pre-trained datasets, models like DeiT and SETR propose training strategies for better performance on smaller datasets. Swin Transformer introduces a hierarchical structure and window-shifting mechanism, achieving state-of-the-art performance in various computer vision tasks. Swin Unet combines Swin Transformer blocks with U-shaped encoder-decoder architecture for robust medical image segmentation. Transformer-based models have also been applied successfully in 3D medical segmentation tasks.

### Self-Attention/Transformer to Complement CNN

Self-attention mechanisms have been integrated into CNN-based models to enhance network performance. Models like Attention U-Net and ASCU-Net integrate attention mechanisms into U-Net for more effective medical image segmentation. However, limitations in learning global and remote semantic information persist. Hybrid CNN-Transformer models have emerged to combine the strengths of both architectures:

- TransUNet: Employs a two-stage encoder structure (CNN to Transformer) for medical image segmentation.
- TransFuse: Uses a parallel approach to combine Transformer and CNN, with a BiFusion module to fuse both features.
- DS-TransUNet: Combines CNN and two scales of Swin Transformer as the encoder and decoder.
- PCAT-UNet: Combines U-shaped network-based Transformer with convolutional branching for global dependencies and feature details.
- X-Net: Implements an X-shaped network structure with dual encoding-decoding combining Transformer and CNN.
- coTr: Utilizes CNN for feature extraction and an efficient deformable Transformer for modeling remote dependencies in 3D medical image segmentation.

## Proposed Method

There are different components:

- **Overall Architecture Design**: CoTrFuse's architecture includes two parallel branches, namely the Transformer Branch and the CNN Branch, for feature extraction. The extracted features from these branches are fused using a module called Swin Transformer and CNN Fusion (STCF). After fusion, the multi-level feature maps are passed through the skip connection and Decoder block for segmentation.

- **EfficientNet Block**: The EfficientNet block is used for feature extraction in the CNN Branch. It comprises multiple MBConvBlocks, including convolution, batch normalization, dropout, Swish activation, and a Squeeze and Excitation block for advanced feature capture. The compound scaling method is employed to optimize model parameters effectively.

- **Swin Transformer Block**: Swin Transformer is introduced as an efficient *alternative* to the standard multi-head self-attention module used in Transformers. It includes the windows multi-head self-attention module (W-MSA) and the shifted windows multi-head self-attention module (SW-MSA) to enhance performance. These modules facilitate information exchange between different windows within a feature map.

- **Swin Transformer and CNN Fusion Module (STCF)**: To effectively fuse features from the Transformer and CNN branches, a novel module called STCF is proposed. It leverages spatial attention mechanisms (CBAM blocks) to exploit spatial relationships between feature maps. The STCF module consists of spatial attention, channel attention, and feature recalibration processes, ultimately combining features from both branches to achieve improved segmentation performance.

## Experiments and results

### Datasets

- ISIC 2017 dataset
- COVID-QU-Ex

### Evaluation metrics

Several metrics are used to evaluate segmentation performance, including Dice coefficient, mean Intersection over Union (mIoU), Precision, Recall, F1-score, and Pixel Accuracy (PA). These metrics are calculated based on true-positive (TP), true-negative (TN), false-positive (FP), and false-negative (FN) values.

### Implementation

- CoTrFuse is implemented using Python 3.8 and PyTorch 1.8.1.
- Training and testing are performed on an Ubuntu 20.04 system with an Nvidia RTX 3090 GPU. (better by far than my setup sadly)
- Data augmentation techniques are applied to enhance model generalization.
- Input image sizes are set to 512x512 for ISIC-2017 and 224x224 for COVID-QU-Ex.
- Swin Transformer and EfficientNet block parameters are initialized using weights pretrained on ImageNet.
- Training details such as optimizer, learning rate, batch size, and epochs are provided for both datasets.

### Experimental results

Very competitive. Often achieved best performances. 

### Ablation study

the CoTrFuse model demonstrates superior segmentation performance on medical image datasets while maintaining computational efficiency. It leverages a combination of CNN and Transformer architectures and introduces innovative fusion mechanisms to achieve state-of-the-art results in medical image segmentation tasks