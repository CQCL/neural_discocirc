import tensorflow as tf

class ValidationAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, get_accuracy_fn, interval=1):
        super(ValidationAccuracy, self).__init__()
        self.get_accuracy_fn = get_accuracy_fn
        self.interval = interval


    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.interval == 0 and self.model.validation_dataset:
            score = self.get_accuracy_fn(self.model, self.model.validation_dataset)
            tf.print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            tf.summary.scalar('validation accuracy', data=score, step=epoch)
