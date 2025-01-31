
from sklearn.model_selection import train_test_split
from Transformer_Latent import create_cnn_transformer_model
import numpy as np

labels = np.load('labels.npy')

images = np.load('images.npy')

# First, create a train/test split
X_train_images, X_test_images, y_train_labels, y_test_labels = train_test_split(
    images, labels, test_size=0.4, random_state=42  # 60% train, 40% temp
)

final_model = create_cnn_transformer_model(
  input_shape=(128, 128, 1),  # Assuming input shape
  latent_dim=256,  # Latent space dimension
  sequence_length=16,  # Sequence length for Transformer
  num_classes=12,  # Number of output classes
  cnn_filters=[32,64,128],
  dropout_rate=0.5,
  num_heads=12,
  dense_dim=256  # Feed-forward layer dimension in Transformer
  )

                        # Compile the model
final_model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
   metrics=['accuracy'])


history = final_model.fit(
    X_train_images, y_train_labels,
    epochs=100,
    batch_size=32
)

test_loss, test_acc = final_model.evaluate(X_test_images, y_test_labels)

print(f"The best test accuracy: {test_acc} and {test_loss}")