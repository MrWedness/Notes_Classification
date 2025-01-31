def get_img_array(img_path, target_size):

    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img = img.resize(target_size)  # Resize to target dimensions
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array


def apply_heatmap_for_transformer(model, img_array, target_size):

    # Target the CNN_Encoder layer
    cnn_encoder_layer = model_trans.get_layer("CNN_Encoder")
    cnn_encoder_model = tf.keras.Model(model_trans.input, cnn_encoder_layer.output)

    # Define the classifier model
    transformer_classifier = model_trans.get_layer("Transformer_Classifier")
    classifier_input = tf.keras.Input(shape=cnn_encoder_layer.output.shape[1:])
    x = classifier_input
    x = transformer_classifier(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Get CNN_Encoder activations
        cnn_encoder_output = cnn_encoder_model(img_array)
        tape.watch(cnn_encoder_output)

        # Compute predictions
        preds = classifier_model(cnn_encoder_output)
        top_pred_index = tf.argmax(preds[0])  # Top predicted class index
        top_class_channel = preds[:, top_pred_index]  # Target class channel

    # Compute gradients of the target class w.r.t. CNN_Encoder output
    grads = tape.gradient(top_class_channel, cnn_encoder_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # Pool the gradients across spatial dimensions

    # Weight the feature maps by the pooled gradients
    cnn_encoder_output = cnn_encoder_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        cnn_encoder_output[:, :, i] *= pooled_grads[i]

    # Generate heatmap by averaging over all channels
    heatmap = np.mean(cnn_encoder_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]

    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]

    # Overlay the heatmap onto the original image
    img = img_array[0].squeeze()  # Remove batch and channel dimensions
    img = cv2.resize(img, target_size)  # Resize to match original dimensions
    img = np.uint8(255 * img)

    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]  # Colormap for the heatmap
    jet_heatmap = jet_colors[heatmap]  # Map heatmap values to colors
    jet_heatmap = tf.keras.utils.img_to_array(tf.keras.utils.array_to_img(jet_heatmap))

    # Superimpose the heatmap with the original image
    superimposed_img = 0.4 * jet_heatmap + np.expand_dims(img, axis=-1)
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Display the superimposed image
    plt.axis("off")
    plt.imshow(superimposed_img)
    plt.show()
