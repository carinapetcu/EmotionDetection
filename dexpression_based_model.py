import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model

from data_loading import DataGenerator


def featEx_block(x, number):
    conv_a = Conv2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', name=f'conv_{number}a')(x)
    conv_b = Conv2D(208, (3, 3), strides=(1, 1), activation='relu', name=f'conv_{number}b')(conv_a)
    maxpool_a = MaxPool2D(3, strides=(1, 1), name=f'maxpool_{number}a')(x)
    conv_c = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', name=f'conv_{number}c')(maxpool_a)
    concat = Concatenate(axis=3, name=f'concat_{number}')([conv_b, conv_c])
    return MaxPool2D((3, 3), strides=(2, 2), name=f'maxpool_{number}b')(concat)


def build_resnet_dexpression_based(input_shape):
    inputs = Input(shape=input_shape)
    conv_1 = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', name='conv_1')(inputs)
    maxpool_1 = MaxPool2D((3, 3), strides=(2, 2), name='maxpool_1')(conv_1)
    x = BatchNormalization()(maxpool_1)

    maxpool_2 = featEx_block(x, 2)
    maxpool_3 = featEx_block(maxpool_2, 3)
    maxpool_4 = featEx_block(maxpool_3, 4)

    net = Flatten()(maxpool_4)
    net = Dense(1024, activation='relu', name='dense1')(net)
    net = Dropout(0.5)(net)
    net = Dense(7, activation='softmax', name='predictions')(net)

    return Model(inputs=inputs, outputs=net, name='deXpression_based')


def train(input_shape, num_classes, train_datagen, val_datagen, num_epochs=5):
    model = build_resnet_dexpression_based(input_shape)
    model.summary()

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(train_datagen, validation_data=val_datagen, epochs=num_epochs)
    model.save(filepath='model_dexpression_based_4.h5')

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    plt.figure()
    plt.plot(history.epoch, training_loss, 'r', label='Training loss')
    plt.plot(history.epoch, validation_loss, 'b--', marker='o', label='Validation loss')
    plt.title('Loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model


if __name__ == '__main__':
    train_generator = DataGenerator('./KDEF/train', 32, (224, 224, 3), 7)
    test_generator = DataGenerator('./KDEF/test', 32, (224, 224, 3), 7, 31, 36)

    model = train((224, 224, 3), 3, train_generator, test_generator, 30)


