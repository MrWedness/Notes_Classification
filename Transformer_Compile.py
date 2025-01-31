def Trans_Compile(filters, dropout_rates, attention_heads, dope, LatentDim):

    results = []

    best_test_acc = 0
    best_Filter = None
    best_do_rate = 0
    best_att_head = 0
    best_latent_dim = 0
    best_dense_dim = 0
    best_model = None

    # Iterate over all combinations of hyperparameters
    for Filt in filters:
        for do in dropout_rates:
            for att in attention_heads:
                for dense_dim in dope:
                    for latent_dim in LatentDim:
                        # Clear session to avoid memory leaks
                        clear_session()

                        # Create the CNN-Transformer model
                        model = create_cnn_transformer_model(
                            input_shape=(128, 128, 1),  # Assuming input shape
                            latent_dim=latent_dim,  # Latent space dimension
                            sequence_length=16,  # Sequence length for Transformer
                            num_classes=12,  # Number of output classes
                            cnn_filters=Filt,
                            dropout_rate=do,
                            num_heads=att,
                            dense_dim=dense_dim  # Feed-forward layer dimension in Transformer
                        )

                        # Compile the model
                        model.compile(optimizer='adam',
                                      loss='sparse_categorical_crossentropy',
                                      metrics=['accuracy'])

                        # Early stopping callback
                        early_stopping = EarlyStopping(
                            monitor='val_accuracy',
                            patience=3,
                            restore_best_weights=True,
                            mode='max'
                        )

                        # Train the model
                        history = model.fit(
                            train_images, train_labels,
                            epochs=50,
                            batch_size=16,
                            validation_data=(val_images, val_labels),
                            callbacks=[early_stopping],
                            verbose=1
                        )

                        # Evaluate on the test set
                        test_loss, test_acc = model.evaluate(test_images, test_labels)
                        print(f"Test accuracy with filters {Filt}, dropout {do}, "
                              f"attention heads {att}, dense_dim {dense_dim}, latent_dim {latent_dim}: {test_acc}")

                        # Append results
                        results.append({
                            'filters': Filt,
                            'dropout_rate': do,
                            'attention_heads': att,
                            'dense_dim': dense_dim,
                            'latent_dim': latent_dim,
                            'test_loss': test_loss,
                            'test_accuracy': test_acc,
                        })

                        # Update best parameters if current model performs better
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            best_Filter = Filt
                            best_do_rate = do
                            best_att_head = att
                            best_latent_dim = latent_dim
                            best_dense_dim = dense_dim
                            best_model = model

    return best_test_acc, best_Filter, best_do_rate, best_att_head, best_latent_dim, best_dense_dim, results, best_model, history