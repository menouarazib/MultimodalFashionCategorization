import numpy as np
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Concatenate, BatchNormalization, Dense, Dropout


class MultiModelClassifier(Model):
    def __init__(self, num_classes: int, l1_lambda: int = None):
        super().__init__()
        self.num_classes = num_classes
        if l1_lambda is None:
            self.dense1 = Dense(units=512, activation="relu")
        else:
            self.dense1 = Dense(units=512, activation="relu", kernel_regularizer=regularizers.l2(l1_lambda))
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(units=256, activation="relu")
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(units=64, activation="relu")
        self.dropout3 = Dropout(0.1)
        self.outputs = Dense(self.num_classes, activation='softmax')
        self.batch_norm = BatchNormalization()
        self.concat = Concatenate()

    def call(self, inputs):
        resnet_inputs, bert_inputs = inputs
        x = self.concat([resnet_inputs, bert_inputs])
        x = self.batch_norm(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        return self.outputs(x)

    def __str__(self):
        return f'MultiModelClassifier with {self.num_classes} classes'


class HierarchicalMultiModelClassifier(Model):
    def __init__(self, num_classes_level1: int, num_classes_level2: int, num_classes_level3: int):
        super().__init__()
        self.num_classes_level1 = num_classes_level1
        self.num_classes_level2 = num_classes_level2
        self.num_classes_level3 = num_classes_level3
        self.dense1 = Dense(units=512, activation="relu")
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(units=256, activation="relu")
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(units=64, activation="relu")
        self.dropout3 = Dropout(0.1)
        self.batch_norm = BatchNormalization()
        self.concat = Concatenate()
        # Output layers for each level
        self.output_level1 = Dense(self.num_classes_level1, activation='softmax')
        self.output_level2 = Dense(self.num_classes_level2, activation='softmax')
        self.output_level3 = Dense(self.num_classes_level3, activation='softmax')

    def call(self, inputs):
        resnet_inputs, bert_inputs = inputs
        x = self.concat([resnet_inputs, bert_inputs])
        x = self.batch_norm(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        return [self.output_level1(x), self.output_level2(x), self.output_level3(x)]

    def __str__(self):
        return f'MultiModelClassifier with {self.num_classes_level1, self.num_classes_level2, self.num_classes_level3} classes'

