# Get the untrained model
# All the other vars should be taken up as a part of your own code. 
# The emphasis in this script is on learning to write the callback with arguments.
model = create_model()


class EvaluateEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        scores = self.model.evaluate(x, y, verbose=False)
        if scores[1] >=0.80 and logs.get('accuracy') >= 0.80:
          print("\nStopping training as accuracy reached.. ")
          self.model.stop_training = True

# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks = [EvaluateEpochEnd(validation_generator)])