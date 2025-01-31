# Enable mixed precision training

def model_compile(filters, dropout_rate, learning_rates, momentum, L1, L2, kernelSize, Opt):

  results = []

  best_test_acc = 0

  best_Filer1 = 0

  best_Filter2 = 0

  best_Filter3 = 0

  best_Filter4 = 0

  best_do_rate = 0

  best_lr_rate = 0

  best_momentum = 0

  best_L1 = 0

  best_L2 = 0

  best_ks = 0

  best_optimiser = None

  for FLNo, Filt in enumerate(filters):
    for opt_class in Opt:
      for ks in kernelSize:
        for lr in learning_rates:
            for do in dropout_rate:
              for mo in momentum:
                for L_1 in L1:
                  for L_2 in L2:
                    clear_session()

                    # Set the policy to mixed precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)

                    model = create_model(Filt[0], Filt[1], Filt[2], Filt[3], ks, do, L_1, L_2)

                    if opt_class == tf.keras.optimizers.SGD:
                      opt = opt_class(learning_rate=lr, momentum=mo)
                    else:
                      opt = opt_class(learning_rate=lr)

                    model.compile(optimizer=opt,
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

                    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, mode='max')

                    history = model.fit(train_images, train_labels,
                                  epochs=50,
                                 batch_size=16,
                                 validation_data=(val_images, val_labels),
                                 callbacks=[early_stopping])

                    test_loss, test_acc = model.evaluate(test_images, test_labels)
                    print(f"Test accuracy for learning rate {lr} and momentum {mo}: {test_acc}")

                    results.append({
                      'optimizer': opt_class,
                      'learning_rate': lr,
                      'momentum': mo,
                      'test_loss': test_loss,
                      'test_accuracy': test_acc,
                      'filters': [Filt[0], Filt[1], Filt[2], Filt[3]],
                      'dropout_rate': do,
                      'L1': L_1,
                      'L2': L_2,
                      'kernel_size': ks
                      })

                    if test_acc > best_test_acc:
                      best_test_acc = test_acc
                      best_Filter = [Filt[0], Filt[1], Filt[2], Filt[3]]
                      best_lr_rate = lr
                      best_do_rate = do
                      best_momentum = mo
                      best_L1 = L_1
                      best_L2 = L_2
                      best_ks = ks

  return best_test_acc, best_Filter, best_do_rate, best_lr_rate, best_momentum, best_L1, best_L2, best_ks, best_optimiser, history, results, model