MNIST Handwritten Digit Classification using ANN
Project Overview

This project implements a fully connected Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset. The dataset is provided as CSV files containing 28x28 grayscale images of digits (0–9).

The goal of this project is to train a neural network to accurately predict digits and visualize its performance through plots, confusion matrices, and sample predictions.

Dataset

Training Data: mnist_train.csv (60,000 images)

Testing Data: mnist_test.csv (10,000 images)

Each row: first column is the label, remaining 784 columns are pixel values (28x28 flattened).

Features

Data Visualization: Display sample images from the dataset.

Preprocessing: Normalize pixel values to [0,1].

Model Architecture:

Dense 512 → ReLU → Dropout 0.3

Dense 256 → ReLU → Batch Normalization → Dropout 0.2

Dense 128 → ReLU

Dense 10 → Softmax

Training: 15 epochs, batch size 128, 20% validation split.

Evaluation: Test accuracy, confusion matrix, classification report.

Visualization: Accuracy & loss curves, predicted vs true labels.

Requirements

Python 3.x and the following packages:

pandas

numpy

tensorflow

matplotlib

seaborn

scikit-learn

Install packages using:

pip install pandas numpy tensorflow matplotlib seaborn scikit-learn

Usage

Clone this repository:

git clone <your-repo-url>


Place mnist_train.csv and mnist_test.csv in the project directory.

Run the main script:

python app.py


Outputs:

Training & validation accuracy/loss curves

Final test accuracy

Confusion matrix heatmap

Classification report

Sample predicted vs true images

Results

Final Test Accuracy: ~98%

The model performs well on unseen test data and demonstrates strong predictive capability for handwritten digits.

Project Structure
MNIST_Project/
│
├─ app.py                # Main Python script
├─ mnist_train.csv       # Training data
├─ mnist_test.csv        # Testing data
└─ README.md             # Project documentation

License

This project is open-source and available under the MIT License.
