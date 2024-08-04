
# CIFAR-10 Image Classification Project

## Project Overview

This project focuses on classifying images from the CIFAR-10 dataset using various deep learning techniques. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The project involves data preprocessing, building and training models, implementing an autoencoder for denoising, and applying transfer learning techniques to improve accuracy.

## Project Structure

- **Data Preprocessing**: Loading the CIFAR-10 dataset and performing normalization and augmentation.
- **Model Building**:
  - **Autoencoder**: Implementing an autoencoder for image denoising.
  - **CNN Model**: Building a Convolutional Neural Network (CNN) for classification.
  - **VGG16 Model**: Applying transfer learning using the VGG16 model.
- **Training and Evaluation**: Training the models, fine-tuning hyperparameters, and evaluating the performance.
- **Results Visualization**: Visualizing the reconstructed images from the autoencoder and the classification results.

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

1. **Clone the repository**:

```bash
git clone https://github.com/SandyHedia/cifar10_project.git
cd cifar10_project
```

2. **Run the Jupyter Notebook**:

Open the `cifar10_project.ipynb` file in Jupyter Notebook or Jupyter Lab and run the cells to execute the project.

3. **Training the Models**:

Follow the steps in the notebook to train the autoencoder, CNN, and VGG16 models.

## Results

- The **autoencoder** successfully denoises images from the CIFAR-10 dataset.
- The **CNN model** achieves significant accuracy in classifying CIFAR-10 images.
- The **VGG16 model** with transfer learning further improves the classification accuracy, achieving a final accuracy of 84.28%.

## Visualizations

The notebook includes visualizations for:

- Original and reconstructed images using the autoencoder.
- Accuracy and loss plots for the training and validation phases.
- Confusion matrix for the classification results.

## Conclusion

This project demonstrates the effectiveness of deep learning models in image classification tasks. The use of autoencoders for denoising and transfer learning with pre-trained models significantly enhances the performance on the CIFAR-10 dataset.

## Future Work

- Experiment with different architectures and hyperparameters to further improve accuracy.
- Explore other pre-trained models for transfer learning.
- Implement additional data augmentation techniques to enhance the dataset.

## Acknowledgements

- The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research (CIFAR).
- The VGG16 model is developed by the Visual Geometry Group (VGG) at the University of Oxford.
