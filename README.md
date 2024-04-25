# UNet Coffee Segmentation

This repository contains the code for training and evaluating a UNet model for coffee leaf rust segmentation. The model is trained using the COFFEE dataset and can be used to segment coffee leaf rust from images, and many other datasets.

## Usage

## Dataset Description

The COFFEE dataset is organized into three main directories: train, val, and test. Each directory contains subdirectories for images and masks. The images are in JPG format, and their corresponding masks are in PNG format. The masks are binary images where the regions of interest are represented as white (pixel value 255) and the background as black (pixel value 0). the dimensions used are 224*224px.
### Prerequisites

- Python 3.x
- TensorFlow
- scikit-image
- matplotlib

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/unet-coffee-segmentation.git

2. **Navigate to the repository directory:**

   ```bash
   cd unet-coffee-segmentation

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

## Training the Model

To train the UNet model, run the train.py script:
    ```bash
    python train.py

This script will load the COFFEE dataset, train the model, and save the trained model weights to a file named modelunetcoffee.h5.
![image](https://github.com/FouadHellal/U-net-for-plant-segmentation/assets/113594352/ba940da7-9e0d-479f-895d-da7ee0c4d4be)

## Evaluating the Model

To evaluate the trained model on the test dataset, run the evaluate.py script:

`python evaluate.py`

This script will load the trained model, evaluate its performance on the test dataset, and print the test loss and accuracy.
# Examples of some masks :
**Coffee dataset :**

![image](https://github.com/FouadHellal/U-net-for-plant-segmentation/assets/113594352/a63acb6c-7f39-459f-80a0-39516365273f)

![image](https://github.com/FouadHellal/U-net-for-plant-segmentation/assets/113594352/196880d9-aa4f-4897-b3f1-a4b27f3a5ba3)

**Plant village (tomato) dataset :**

![image](https://github.com/FouadHellal/U-net-for-plant-segmentation/assets/113594352/ecef34e0-419f-41a2-b464-0fb1ea2e013f)

![image](https://github.com/FouadHellal/U-net-for-plant-segmentation/assets/113594352/3cd3e12b-dcfc-4ae0-a0f5-72c2d5951691)


## Making Predictions

To make predictions using the trained model, you can use the predict.py script:

    
    python predict.py --image_path /path/to/image.jpg
    

Replace /path/to/image.jpg with the path to the image for which you want to generate a segmentation mask.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


Feel free to reach out to us if you have any questions or suggestions!
