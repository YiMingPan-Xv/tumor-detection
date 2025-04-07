import numpy as np
import os
import pandas as pd
from keras import models, layers
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from make_dataset import initialize
from preprocess import resize_and_gray, augment

img_size = (128, 128)

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_df():

    df = initialize(force=False)

    df['rawimage'] = df['path'].apply(lambda x: np.array(Image.open(x)))
    df['image'] = resize_and_gray(df['rawimage'], img_size)
    df.drop(columns=['path', 'rawimage'], inplace=True)

    augmented_data = augment(df['image'], df['label'])

    aug_df = pd.DataFrame(augmented_data, columns=['image', 'label'])

    upd_df = pd.concat([df, aug_df], axis=0, ignore_index=True)

    return upd_df

def test_model(model, X_test, y_test):
    y_pred = [round(pred[0]) for pred in model.predict(X_test)]
    print(y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"F1 score: {f1}")
    print(f"Confusion Matrix: {cm}")



if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent # assuming src is one level deep
    os.chdir(project_dir)

    df = get_df()
    X_train, X_test, y_train, y_test = train_test_split(df['image'], df['label'], 
                                                        test_size=0.3, random_state=42, shuffle=True)
    
    X_train = np.array(X_train.tolist(), dtype='float32') / 255.0
    X_test = np.array(X_test.tolist(), dtype='float32') / 255.0
    X_train = X_train.reshape(-1, img_size[0], img_size[1], 1)
    X_test = X_test.reshape(-1, img_size[0], img_size[1], 1)

    if os.path.exists("model/tumormodel.keras"):
        i = input("Existing model found:\nTrain a new model? (Y/n)")
        if i.lower() == 'n':
            model = models.load_model("model/tumormodel.keras")
            test_model(model, X_test, y_test)
            os._exit(0)

    model = build_model((img_size[0], img_size[1], 1))
    model.fit(X_train, y_train, epochs=200, batch_size=32)

    test_model(model, X_test, y_test)

    i = input("Save the model? (y/N)")
    if i.lower() == 'y':
        model.save("model/tumormodel.keras")

    