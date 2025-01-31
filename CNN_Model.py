from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, RandomFlip, RandomRotation
import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model


def create_model(Filter1, Filter2, Filter3, Filter4, kernelSize, Dropout_Rate, L1, L2):


    # Input layer based on the input shape
    inputs = Input(shape=(128, 128, 1))

    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1)
    ])
    x = data_augmentation(inputs)

    # Feature extraction layers
    x = Conv2D(Filter1, kernel_size=3, activation='relu', padding='valid', strides=(1, 1), kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = MaxPooling2D(pool_size=2, strides=(1, 1))(x)

    x = Conv2D(Filter2, kernel_size=3, activation='relu', padding='valid', strides=(1, 1), kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = MaxPooling2D(pool_size=2, strides=(1, 1))(x)

    x = Conv2D(Filter3, kernel_size=3, activation='relu', padding='valid', strides=(1, 1), kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = MaxPooling2D(pool_size=2, strides=(1, 1))(x)

    x = Conv2D(Filter4, kernel_size=3, activation='relu', padding='valid', strides=(1, 1), kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = MaxPooling2D(pool_size=2, strides=(1, 1))(x)

    # Classification layers
    x = Dropout(Dropout_Rate)(x)  # Regularization to prevent overfitting
    x = Flatten()(x)  # Convert 2D feature maps to 1D vector

    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)  # Fully connected layer with 128 neurons
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)  # Fully connected layer with 256 neurons
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)  # Fully connected layer with 512 neurons

    # Output layer
    outputs = Dense(12, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model
