import logging
import os
import pickle

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from model import MultiModelClassifier, HierarchicalMultiModelClassifier
from utils import load_resnet, load_bert_tokenize_model, generate_embeddings, plot_history

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_saved_embeddings = True
dir_name = "models"


def load_and_preprocess_data():
    logger.info("Loading and preprocessing data...")
    # Set a seed for the random number generator
    np.random.seed(0)
    dataset = load_dataset("ashraq/fashion-product-images-small")['train']
    labels_level_1 = dataset['masterCategory']
    labels_level_2 = dataset['subCategory']
    labels_level_3 = dataset['articleType']
    # Create a label encoder

    # Fit the encoder to the labels
    classes_level_1 = LabelEncoder().fit_transform(labels_level_1)
    classes_level_2 = LabelEncoder().fit_transform(labels_level_2)
    classes_level_3 = LabelEncoder().fit_transform(labels_level_3)

    return dataset, (classes_level_1, classes_level_2, classes_level_3)


def train_model(train_dataset, classes, resnet_model, bert_model, tokenizer):
    logger.info("Training model...")
    if not use_saved_embeddings:
        image_embeddings_array, description_embeddings_array = generate_embeddings(train_dataset=train_dataset,
                                                                                   resnet_model=resnet_model,
                                                                                   bert_model=bert_model,
                                                                                   tokenizer=tokenizer)
        # Save the embeddings
        np.save('image_embeddings_array.npy', image_embeddings_array)
        np.save('description_embeddings_array.npy', description_embeddings_array)
    else:
        image_embeddings_array, description_embeddings_array = np.load("image_embeddings_array.npy"), np.load(
            "description_embeddings_array.npy")

    name = "best_model"
    if len(classes) == 3:
        name = "hierarchical_" + name
        model = HierarchicalMultiModelClassifier(num_classes_level1=len(np.unique(classes[0])),
                                                 num_classes_level2=len(np.unique(classes[1])),
                                                 num_classes_level3=len(np.unique(classes[2])))
    else:
        model = MultiModelClassifier(num_classes=len(np.unique(classes[0])))

    # multi_model_classifier = MultiModelClassifier(num_classes=len(np.unique(train_class_numbers)))

    model.compile(optimizer=Adam(1e-3),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Create a callback that saves the model's weights
    checkpoint = ModelCheckpoint(filepath=os.path.join(dir_name, name),
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,  # Save the entire model, not just the weights
                                 mode='max')

    callbacks_list = [checkpoint]

    # Assuming classes is a list of your targets: [classes_level1, classes_level2, classes_level3]
    if len(classes) == 3:
        x_train_image, x_valid_image, y_train_1, y_valid_1, y_train_2, y_valid_2, y_train_3, y_valid_3 = train_test_split(
            image_embeddings_array, classes[0], classes[1], classes[2], test_size=0.2,
            random_state=42
        )

        print(type(y_train_1), y_train_1.shape)
        print(type(y_train_2), y_train_2.shape)
        print(type(y_train_3), y_train_3.shape)

        print(type(y_valid_1), y_valid_1.shape)
        print(type(y_valid_2), y_valid_2.shape)
        print(type(y_valid_3), y_valid_3.shape)

        y_train = [y_train_1, y_train_2, y_train_3]
        y_valid = [y_valid_1, y_valid_2, y_valid_3]

    else:
        x_train_image, x_valid_image, y_train, y_valid = train_test_split(
            image_embeddings_array, classes[0], test_size=0.2, random_state=42
        )

    x_train_desc, x_valid_desc = train_test_split(description_embeddings_array, test_size=0.2, random_state=42)

    print(type(y_train), type(y_valid))

    if len(classes) == 3:
        history = model.fit([x_train_image, x_train_desc], y_train, epochs=12,
                            validation_data=([x_valid_image, x_valid_desc], y_valid))
    else:
        history = model.fit([x_train_image, x_train_desc], y_train, epochs=12,
                            validation_data=([x_valid_image, x_valid_desc], y_valid),
                            callbacks=callbacks_list)

    return model, history


def main():
    logger.info("Starting program...")
    # Check if the directory already exists
    if not os.path.exists(dir_name):
        # Create your directory
        os.makedirs(dir_name)
        logger.info(f"'{dir_name}' directory created.")
    else:
        logger.info(f"'{dir_name}' directory already exists.")

    # Load pretrained models
    resnet_model = load_resnet()
    bert_model, tokenizer = load_bert_tokenize_model()

    dataset, classes = load_and_preprocess_data()
    multi_model_classifier, history = train_model(dataset, classes, resnet_model, bert_model,
                                                  tokenizer)

    history_path = os.path.join(dir_name, 'history.pickle')
    # Save the history object as a pickle file
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    name = "multi_model_classifier"
    if len(classes) == 3:
        name = "hierarchical_multi_model_classifier"

    model_path = os.path.join(dir_name, name)
    multi_model_classifier.save(model_path)
    logger.info("Program finished!")

    # Plot Losses and Accuracies
    plot_history(history.history)


if __name__ == "__main__":
    main()
