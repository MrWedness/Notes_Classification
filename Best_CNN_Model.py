#I fit this model with the best hyperparamters this was to get a full val acc/loss vs training epoch graph
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.4, random_state=42  
)
model = create_model(64, 64, 32, 128, 3, 0, 0, 0)

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.5)

model.compile(optimizer=opt,
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

history = model.fit(
    train_images, train_labels,
    epochs=50,
    batch_size=16)
 

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"The best test accuracy: {test_acc} and {test_loss}")