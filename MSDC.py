import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Layer, Concatenate, Add, Activation, Lambda, Multiply

class DeformableConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.offset_conv = Conv2D(
            filters=2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            padding='same'
        )
        self.kernel_conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same'
        )
        super(DeformableConv2D, self).build(input_shape)

    def call(self, inputs):
        offsets = self.offset_conv(inputs)
        offsets = Lambda(lambda x: tf.keras.activations.tanh(x))(offsets)  # Activation function for offsets

        # Generate sampling grid
        grid = tf.meshgrid(tf.range(inputs.shape[1]), tf.range(inputs.shape[2]))
        grid = tf.stack(grid, axis=-1)
        grid = tf.cast(grid, dtype=tf.float32)

        offsets = grid + offsets

        # Bilinear interpolation using gathered_nd
        deformed_inputs = tf.gather_nd(inputs, tf.cast(offsets, dtype=tf.int32))

        # Apply convolution with the deformed inputs
        output = self.kernel_conv(deformed_inputs)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

def LoadData():
    (x_train, _), (x_test, _) = mnist.load_data()
    # print(x_train.shape)    # (60000, 28, 28)
    # print(x_test.shape)     # (10000, 28, 28)
    return x_train, x_test

def Preprocess(x_train, x_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

def MSDC(x, filters, kernel_size):
    # Create a deformable convolution layer
    deformable_conv = DeformableConv2D(filters=filters, kernel_size=kernel_size)

    # Pass the input through the deformable convolution layer
    output = deformable_conv(x)

    return output

def RB(x, filters):
    shortcut = x
    newF = filters // 2

    x = Conv2D(newF, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(newF, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (1, 1), activation='relu', padding='same')(x)

    x = Add()([x, shortcut])
    return x

def MSDC_RB(x, filters):
    shortcut = x
    newF = filters // 2

    x = Conv2D(newF, (1, 1), activation='relu', padding='same')(x)
    x = MSDC(x, filters=newF, kernel_size=(3, 3))
    x = Conv2D(filters, (1, 1), activation='relu', padding='same')(x)

    x = Add()([x, shortcut])
    return x

def MSDSAM(x, filters):
    xs = [x, x, x]

    xs[1] = RB(xs[1], filters=filters)
    xs[1] = RB(xs[1], filters=filters)
    xs[1] = RB(xs[1], filters=filters)

    xs[2] = MSDC_RB(xs[2], filters=filters)
    xs[2] = MSDC_RB(xs[2], filters=filters)
    xs[2] = MSDC_RB(xs[2], filters=filters)
    xs[2] = Conv2D(filters, (1, 1), activation='relu', padding='same')(xs[2])
    xs[2] = Activation('sigmoid')(xs[2])

    x3 = Multiply()([xs[1], xs[2]])
    x = Add()([xs[0], x3])
    return x

def DMMSD(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = MSDC(x, filters=filters, kernel_size=(3, 3))

    shortcut = Conv2D(filters, (1, 1), strides=2, activation='relu', padding='same')(shortcut)

    x = Add()([x, shortcut])
    return x

def Encoder(input_img, filters=28):
    x = Conv2D(filters, (3, 3), strides=2, activation='relu', padding='same')(input_img)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = MSDSAM(x, filters)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = DMMSD(x, filters)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), strides=2, activation='relu', padding='same')(x)

    x = MSDSAM(x, filters)
    encoded = x

    return encoded

def UMMSD(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = MSDC(x, filters=filters, kernel_size=(3, 3))

    shortcut = Conv2D(filters, (1, 1), activation='relu', padding='same')(shortcut)
    shortcut = UpSampling2D((2, 2))(shortcut)

    x = Add()([x, shortcut])
    return x

def Decoder(encoded, filters=28):
    x = MSDSAM(encoded, filters)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = UMMSD(x, filters=filters)

    x = MSDSAM(x, filters=filters)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x) 

    finalFilters = 1
    x = Conv2D(finalFilters, (3, 3), activation='sigmoid', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = x 

    return decoded    

def rate_distortion_loss(y_true, y_pred):
    # Calculate rate and distortion terms based on y_true and y_pred
    
    def Rate(y_true, y_pred):
        # Assuming rate is proportional to the number of non-zero elements in y_pred
        # You might have a more complex calculation based on your actual compression method
        non_zero_elements = tf.math.count_nonzero(y_pred)
        # print(tf.cast(tf.size(y_pred), tf.int32).dtype)
        rate = non_zero_elements / tf.cast(tf.size(y_pred), tf.int64)
        return rate
    
    def Distortion(y_true, y_pred):
        y_true_transposed = tf.expand_dims(y_true, axis=-1)
        y_true_resized = tf.image.resize(y_true_transposed, (32, 32), method=tf.image.ResizeMethod.BILINEAR)
        y_true_resized = tf.squeeze(y_true_resized, axis=-1)

        mse = tf.reduce_mean(tf.square(tf.cast(y_true_resized, tf.float32) - y_pred))
        return mse

    rate_term = Rate(y_true, y_pred)  # Calculate rate term based on y_true and y_pred
    distortion_term = Distortion(y_true, y_pred)  # Calculate distortion term based on y_true and y_pred

    # lambda
    tradeoff_parameter = 0.0075

    # Calculate the combined loss (trade-off between rate and distortion)
    combined_loss = tf.cast(distortion_term, tf.float32) + tradeoff_parameter * tf.cast(rate_term, tf.float32)
    return combined_loss


def DisplayImages(originals, reconstructed):
    n = 10  # Number of images to display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(originals[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    
    x_train, x_test = LoadData()

    Preprocess(x_train, x_test)
    # print(x_train.shape)    # (60000, 28, 28, 1)

    # Define the autoencoder architecture
    input_img = Input(shape=(28, 28, 1))

    # Encoder
    encoded = Encoder(input_img)

    # Decoder
    decoded = Decoder(encoded)

    # Create the autoencoder model
    autoencoder = Model(input_img, decoded)

    # Compile the autoencoder
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss=rate_distortion_loss)
    # autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

    # Encode and decode the test images
    decoded_imgs = autoencoder.predict(x_test)

    # Display original and reconstructed images
    DisplayImages(originals=x_test, reconstructed=decoded_imgs)