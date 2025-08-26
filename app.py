import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")


print("Training dataframe shape:", train_df.shape)
print("Testing dataframe shape:", test_df.shape)


print("Columns in dataset:", train_df.columns[:10], "...", "and total:", train_df.shape[1])


fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    img = train_df.iloc[i, 1:].values.reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {train_df.iloc[i, 0]}")
    ax.axis("off")
plt.suptitle("Sample MNIST Images from CSV")
plt.show()


x_train = train_df.drop('label', axis=1).values.astype("float32") / 255.0
y_train = train_df['label'].values


x_test = test_df.drop('label', axis=1).values.astype("float32") / 255.0
y_test = test_df['label'].values


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)



def build_ann_model():
    return keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

model = build_ann_model() #bulid ann model




model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)



test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()




plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()




y_pred = model.predict(x_test).argmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




print("\nClassification Report:")
print(classification_report(y_test, y_pred))




plt.figure(figsize=(10, 5))
for i in range(10):
    img = x_test[i].reshape(28, 28)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"P: {y_pred[i]}, T: {y_test[i]}")
    plt.axis("off")
plt.suptitle("Predicted vs True Labels")
plt.show()