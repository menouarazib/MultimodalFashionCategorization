# Import necessary libraries
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import Dataset
from matplotlib import pyplot as plt
from transformers import BertTokenizer, TFBertModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_resnet() -> tf.keras.Model:
    """
    Load the pre-trained ResNet50 model from Keras, excluding the top layer.
    """
    base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    gap = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=gap)
    return model


def preprocess_image(img_pil: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to be compatible with ResNet50.
    """
    img_resized = img_pil.resize((224, 224))
    img_resized_np = np.array(img_resized)
    if len(img_resized_np.shape) == 2:
        img_resized_np = np.stack((img_resized_np,) * 3, axis=-1)
    img_resized_np = np.expand_dims(img_resized_np, axis=0)
    img_resized_np = tf.keras.applications.resnet50.preprocess_input(img_resized_np)
    return img_resized_np


def load_bert_tokenize_model() -> Tuple[TFBertModel, BertTokenizer]:
    """
    Load the pre-trained BERT model and tokenizer from the transformers library.
    """
    model = TFBertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


def generate_embeddings(train_dataset: Dataset, resnet_model: tf.keras.Model, bert_model: TFBertModel,
                        tokenizer: BertTokenizer) -> Tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray]]:
    """
    Generate image and text embeddings for each sample in the training dataset.
    """
    image_embeddings_list = []
    description_embeddings_list = []

    for i, sample in enumerate(train_dataset):
        logger.info(f"Generating Embeddings for item number {i} - {sample['productDisplayName']}")
        # Generate embeddings for the image
        image_pil = sample['image']
        image_pil = preprocess_image(image_pil)
        image_embeddings = resnet_model.predict(image_pil, verbose=0)
        # Generate embeddings for the description
        description = sample['productDisplayName']
        inputs = tokenizer(description, return_tensors='tf')
        outputs = bert_model(inputs)
        embeddings = outputs.last_hidden_state
        description_embeddings = tf.reduce_mean(embeddings, axis=1)
        image_embeddings_list.append(image_embeddings[0])
        description_embeddings_list.append(description_embeddings[0])

    # Convert the lists to numpy arrays
    image_embeddings_array = np.array(image_embeddings_list)
    description_embeddings_array = np.array(description_embeddings_list)

    return image_embeddings_array, description_embeddings_array


def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
