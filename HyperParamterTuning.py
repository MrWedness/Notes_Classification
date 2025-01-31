from sklearn.model_selection import train_test_split
import numpy as np
from CNN_Model import create_model
from Compile_Model import model_compile
import tensorflow as tf
import pandas as pd

labels = np.load('labels.npy')

images = np.load('images.npy')

# First, create a train/test split
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.4, random_state=42  # 60% train, 40% temp
)

# Then, split the temp set into validation and test sets
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42  # 20% val, 20% test
)

filters = [[64,64,32,128], [64, 32, 64, 256]]

dropout_rate = [0, 0.5]

learning_rates = [0.01, 0.1]

momentum = [0.0, 0.5]

L1 = [0, 0.1]

L2 = [0, 0.1]

kernelSize = [3, 5, 7]

Opt = [
    tf.keras.optimizers.SGD]

best_test_acc, best_Filter, best_do_rate, best_lr_rate, best_momentum, best_L1, best_L2, best_ks, best_optimiser, history, results,model = model_compile(filters, dropout_rate, learning_rates, momentum, L1, L2, kernelSize, Opt)

best_results_dict = {'testAccuract': best_test_acc, 'dropOutRate': best_do_rate, 'bestLrRate': best_lr_rate, 'bestMomentum': best_momentum, 'bestL1': best_L1, 'bestL2': best_L2, 'bestKS': best_ks, 'bestOptimiser': best_optimiser}

Results_Table = pd.DataFrame(best_results_dict)

