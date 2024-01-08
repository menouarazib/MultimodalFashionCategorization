import logging
import os
import pickle

import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from model import MultiModelClassifier
from utils import load_resnet, load_bert_tokenize_model, generate_embeddings, plot_history

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_saved_embeddings = False
dir_name = "models"

def load_and_preprocess_data():
    logger.info("Loading and preprocessing data...")
    # Set a seed for the random number generator
    np.random.seed(0)
    dataset = load_dataset("ashraq/fashion-product-images-small")['train']
    labels = dataset['subCategory']

    # Create a label encoder
    le = LabelEncoder()
    # Fit the encoder to the labels
    classes = le.fit_transform(labels)

    return dataset, classes


def train_model(train_dataset, train_class_numbers, resnet_model, bert_model, tokenizer):
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

    multi_model_classifier = MultiModelClassifier(num_classes=len(np.unique(train_class_numbers)))

    multi_model_classifier.compile(optimizer=Adam(1e-3),
                                   loss=SparseCategoricalCrossentropy(),
                                   metrics=['accuracy'])

    # Create a callback that saves the model's weights
    checkpoint = ModelCheckpoint(filepath=os.path.join(dir_name, "best_model"),
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,  # Save the entire model, not just the weights
                                 mode='max')

    callbacks_list = [checkpoint]

    # Fit the model with the callback
    history = multi_model_classifier.fit([image_embeddings_array, description_embeddings_array],
                                         train_class_numbers[:51],
                                         epochs=30,
                                         validation_split=0.2, callbacks=callbacks_list)

    return multi_model_classifier, history


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

    model_path = os.path.join(dir_name, "multi_model_classifier")
    multi_model_classifier.save(model_path)
    logger.info("Program finished.")

    # Plot Losses and Accuracies
    plot_history(history)


if __name__ == "__main__":
    main()
