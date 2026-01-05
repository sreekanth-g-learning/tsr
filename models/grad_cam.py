import tensorflow as tf
import numpy as np
'''
Generates a Grad-CAM heatmap for a given model and input image.
inputs:
- model: A tf.keras Model object.
- img_array: Preprocessed image array of shape (1, height, width, channels).
- last_conv_layer: Name of the last convolutional layer in the model.
outputs:- heatmap: A 2D numpy array representing the Grad-CAM heatmap.
         - pred_index: The index of the predicted class.    

'''
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):
    # 1. create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ]
    )
    # 2. compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    # 3. This is the gradient of the output neuron (top predicted)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(pred_output, conv_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    #  multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    conv_output = conv_output[0]
    #heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy(), int(pred_index)
