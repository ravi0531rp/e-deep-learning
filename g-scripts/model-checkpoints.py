import tensorflow as tf
lrate = 0.001
callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "./path/to/model",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
            save_freq="epoch",
            mode="min",
            period=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=7),
        lrate  
    ]