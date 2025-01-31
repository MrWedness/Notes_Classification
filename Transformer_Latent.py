import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model



class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.dense_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_dim, activation="relu"),
            tf.keras.layers.Dense(self.embed_dim),
        ])
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

# Create a combined CNN-Transformer model function for hyperparameter tuning
def create_cnn_transformer_model(input_shape, latent_dim, sequence_length, num_classes, cnn_filters, dropout_rate, num_heads, dense_dim):

    # CNN Encoder
    def create_cnn_encoder():
      inputs = Input(shape=input_shape)
      x = Conv2D(cnn_filters[0], (3, 3), activation='relu', padding='same')(inputs)
      x = MaxPooling2D((2, 2))(x)
      x = Conv2D(cnn_filters[1], (3, 3), activation='relu', padding='same')(x)
      x = MaxPooling2D((2, 2))(x)
      x = Conv2D(cnn_filters[2], (3, 3), activation='relu', padding='same')(x)
      x = GlobalAveragePooling2D()(x)
      latent_space = Dense(latent_dim, activation='relu')(x)
      model = Model(inputs, latent_space, name="CNN_Encoder")
      return model

    # Transformer Classifier
    def create_transformer_classifier():
        inputs = tf.keras.Input(shape=(sequence_length, latent_dim))
        x = TransformerEncoder(latent_dim, dense_dim, num_heads)(inputs)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs, name="Transformer_Classifier")
        return model


    # Instantiate CNN Encoder
    cnn_encoder = create_cnn_encoder()

    # Instantiate Transformer Classifier
    transformer_classifier = create_transformer_classifier()

    # Combined Model
    image_inputs = Input(shape=input_shape)
    latent_space = cnn_encoder(image_inputs)

    # Prepare latent space for transformer
    flattened_latent = Dense(sequence_length * latent_dim)(latent_space)
    reshaped_latent = Reshape((sequence_length, latent_dim))(flattened_latent)

    outputs = transformer_classifier(reshaped_latent)
    model = Model(inputs=image_inputs, outputs=outputs, name="CNN_Transformer_Model")

    return model
