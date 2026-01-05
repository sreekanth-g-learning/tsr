from tensorflow.keras import layers, models

'''
    build_custom_cnn creates a customizable CNN model.
    Parameters:
    - input_shape: tuple, shape of the input images (height, width, channels), default is (30, 30, 3)
    - num_classes: int, number of output classes,default is 43
    - conv_filters: list of int, number of filters for each Conv2D layer, default is [16, 32, 64, 128]
    - kernel_size: tuple, size of the convolution kernels, default is (3, 3)
    - use_batchnorm: bool, whether to include BatchNormalization layers, default is True
    - use_dropout: bool, whether to include Dropout layer, default is True
    - dropout_rate: float, dropout rate if Dropout is used, default is 0.5
    - dense_units: int, number of units in the dense layer before output, default is 512
    Returns:
    - model: Keras Model instance
'''
#==============================================================
# Customizable CNN Model Builder - 
# Build a CNN with flexible architecture options with parameterized CNN blocks, batch normalization and dropout
#==============================================================
def build_custom_cnn(
    input_shape=(30, 30, 3),
    num_classes=43,
    conv_filters=[16, 32, 64, 128],
    kernel_size=(3, 3),
    use_batchnorm=True,
    use_dropout=True,
    dropout_rate=0.5,
    dense_units=512,
    model_name="custom_cnn"
):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i, filters in enumerate(conv_filters):
        x = layers.Conv2D(filters, kernel_size, padding="valid", activation="relu")(x)
        if i % 2 == 1:  # pooling every 2 conv layers
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            if use_batchnorm:
                x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)

    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=model_name)
    return model
