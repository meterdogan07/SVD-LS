# Multi-class Pneumonia Classification on X-Ray Images using Transfer Learning and Self-Supervised Fine-Tuning

## Overview

Pneumonia is a critical respiratory infection, and early diagnosis is essential for effective treatment. This project explores automated classification of X-ray images into three categories: normal, bacterial pneumonia, and viral pneumonia. We apply several advanced machine learning models and techniques, including:

1. **Transfer Learning with ResNet18**: Training from scratch, full model transfer learning, and fine-tuning the last layer.
2. **Self-Supervised Learning with DINO ViT-B/16**: Fine-tuning a linear classifier on pre-trained transformer features.
3. **Singular Value Decomposition based Least Squares (SVD-LS)**: Leveraging PCA for dimensionality reduction and training a linear classifier on feature vectors extracted by the above models.

## Key Features

- **Dataset**: We use the Chest X-Ray dataset from the Guangzhou Women and Children’s Medical Centre, available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). The dataset consists of labeled X-ray images categorized as healthy, bacterial pneumonia, or viral pneumonia.
- **Preprocessing**: Includes resizing, normalization, adaptive histogram equalization (CLAHE), and channel duplication for grayscale images.
- **Data Augmentation**: Horizontal flipping, rotation, and Gaussian blur applied to balance class distribution during k-fold cross-validation.
- **Models and Training**: Detailed grid-search tuning for hyperparameters and robust training using gradient descent methods (Adam optimizer) and Extreme Learning Machines.

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone ____
    ```

2. Install dependencies:
    ## Dependencies

    The project uses the following libraries and packages:
    
    - Python 3.8+
    - NumPy
    - Matplotlib
    - PIL (Python Imaging Library)
    - tqdm
    - OpenCV (cv2)
    - Torch (PyTorch)
    - Torchvision
    - scikit-learn
    - IPython
    - pickle (standard library)
    - collections (standard library)
    - logging (standard library)
    - sys (standard library)
    - time (standard library)
    - os (standard library)
    - shutil (standard library)
    - math (standard library)
    - random (standard library)
    - copy (standard library)
    
    You can install the necessary packages using the following command:
    
    ```bash
    pip install numpy matplotlib pillow tqdm opencv-python torch torchvision scikit-learn ipython
    ```

3. Run the training scripts:
    ```bash
    python train_model_kfold.py
    ```
    Change the parameter model_no in the 'k_fold_cross_validation' function to try different models, based on the below section.


## Model Configurations and Parameters

This project provides multiple pre-trained models for experimentation. Below are the recommended model configurations, learning rates, and fine-tuning strategies:

### 1. Vision Transformer (ViT-B/16)
**Model Number: `4`**

- **Learning Rate**: `1e-4`
- **Frozen Layers**: All layers except the classification head (`model.heads.head`).
- **Fine-tuning Strategy**: Fine-tune only the last layer. The classification head is replaced with a new `nn.Linear(768, 3)` layer.
- **Output Layer**: `768` input features, `3` output classes.

### 2. DINO (Self-Supervised Vision Transformer, ViT-S/16)
**Model Number: `3`**

- **Learning Rate**: `1e-5`
- **Frozen Layers**: All layers except the custom classification layer (`model.last`).
- **Fine-tuning Strategy**: Fine-tune only the last layer. The model uses a pre-trained DINO backbone with a `nn.Linear(384, 3)` classification layer.
- **Output Layer**: `384` input features, `3` output classes.

### 3. ResNet18 (Pre-trained with Frozen Layers)
**Model Number: `2`**

- **Learning Rate**: `1e-3`
- **Frozen Layers**: All layers except the fully connected layer (`model.fc`).
- **Fine-tuning Strategy**: Fine-tune only the last layer. The fully connected layer is replaced with a new `nn.Linear(512, 3)` layer.
- **Output Layer**: `512` input features, `3` output classes.

### 4. ResNet18 (Fully Fine-tuned)
**Model Number: `1`**

- **Learning Rate**: `1e-3`
- **Frozen Layers**: None (all layers are trainable).
- **Fine-tuning Strategy**: Fully fine-tune the model with the original pre-trained weights.
- **Output Layer**: Default ResNet18 output (`1000` classes), but should be followed by a final classification layer depending on your task.

### 5. ResNet18 (Without Pre-training)
**Model Number: `0` (Default)**

- **Learning Rate**: `1e-2`
- **Frozen Layers**: None (all layers are trainable).
- **Fine-tuning Strategy**: Train from scratch with random initialization.
- **Output Layer**: Default ResNet18 output (`1000` classes).

### General Notes:
- Fine-tuning typically benefits from lower learning rates (e.g., `1e-4` to `1e-5`) compared to training from scratch (e.g., `1e-3` to `1e-2`).
- Monitor performance during training and adjust the learning rate if necessary.


## Project Structure

- 'loggers_aug_last': Contains the experiments for the Gradient Descent based models.

- 'train_model_kfold.py': Contains the code to train Gradient Descent based models with data augmentation and k-fold cross validation.

- 'prepare-data.ipynb': Contains the code to create and save the train and test datasets as pickle files. Applies CLAHE and mean, std normalization on the images.

- 'SVD-LS-kfold.ipynb': Contains the code to train SVD-LS based models and perform tests. This notebook also loads and tests the Gradient Descent based models and calculates r-bowker tests on 7 models.


## Results

We evaluated the models using 5-fold cross-validation and reported the test set results based on the best validation F1-score. Detailed training configurations and hyperparameters used can be found in the report.

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", CVPR 2021.
- Huang et al., "Extreme Learning Machine: A New Learning Scheme of Feedforward Neural Networks", Neurocomputing 2006.


## License

This project is licensed under the MIT License.
