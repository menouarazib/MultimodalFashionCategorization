from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, BatchNormalization, Dense, Dropout


class MultiModelClassifier(Model):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.dense1 = Dense(units=512)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(units=256)
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(units=64, activation="relu")
        self.dropout3 = Dropout(0.5)
        self.outputs = Dense(self.num_classes, activation='softmax')
        self.batch_norm = BatchNormalization()
        self.concat = Concatenate()

    def call(self, inputs):
        resnet_inputs, bert_inputs = inputs
        x = self.concat([resnet_inputs, bert_inputs])
        x = self.batch_norm(x)
        """
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        """
        # x = self.dense3(x)
        # x = self.dropout3(x)
        return self.outputs(x)

    def __str__(self):
        return f'MultiModelClassifier with {self.num_classes} classes'
