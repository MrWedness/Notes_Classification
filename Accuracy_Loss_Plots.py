def plot_loss_and_accuracy(histories):
    history_dict = histories.history

    # Get loss and accuracy for training and validation
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]

    # Define epochs range
    epochs = range(1, len(loss) + 1)

    # Plot for loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # First plot (for loss)
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for accuracy
    plt.subplot(1, 2, 2)  # Second plot (for accuracy)
    plt.plot(epochs, acc, 'r-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Display both plots
    plt.tight_layout()
    plt.show()