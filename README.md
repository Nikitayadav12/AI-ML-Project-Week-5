✨ MNIST Handwritten Digit Classification using ANN
📝 Project Overview

This project implements a fully connected Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset.

📌 Goal: Train a neural network to predict digits (0–9) accurately.

🎯 Output: Predictions, accuracy plots, confusion matrix, and sample visualizations.

📂 Dataset

Training Data: mnist_train.csv (60,000 images)

Testing Data: mnist_test.csv (10,000 images)

🖼 Each row: first column = label, remaining 784 columns = pixel values (28x28 flattened).

🔑 Features

📊 Data Visualization: Display sample images from the dataset.

⚡ Preprocessing: Normalize pixel values to [0,1].

🏗 Model Architecture:

Dense 512 → ReLU → Dropout 0.3

Dense 256 → ReLU → Batch Normalization → Dropout 0.2

Dense 128 → ReLU

Dense 10 → Softmax

🏋️ Training: 15 epochs, batch size 128, 20% validation split

📈 Evaluation: Test accuracy, confusion matrix, classification report

🖌 Visualization: Accuracy & loss curves, predicted vs true labels

⚙️ Requirements

Python 3.x and the following packages:

pandas 🐼

numpy 🔢

tensorflow 🤖

matplotlib 📉

seaborn 🌊

scikit-learn 🛠

Install packages via:

pip install pandas numpy tensorflow matplotlib seaborn scikit-learn

🚀 Usage

Clone this repository:

git clone <your-repo-url>


Place mnist_train.csv and mnist_test.csv in the project folder.

Run the script:

python app.py


Outputs:

📊 Training & validation accuracy/loss curves

✅ Final test accuracy

🗂 Confusion matrix heatmap

🧾 Classification report

🖼 Sample predicted vs true images

📊 Results

Final Test Accuracy: ~98% ✅

The model performs well on unseen data and predicts handwritten digits accurately.

🗂 Project Structure
MNIST_Project/
│
├─ app.py                # Main Python script
├─ mnist_train.csv       # Training data
├─ mnist_test.csv        # Testing data
└─ README.md             # Project documentation

📜 License

This project is open-source and available under the MIT License. 🛡
