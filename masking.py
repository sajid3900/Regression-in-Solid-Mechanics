import tensorflow as tf

class SequenceMasking(tf.keras.layers.Layer):
    """Masks a sequence by using a mask value to skip timesteps.
    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).
    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.
    Example:
    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer.
    You want to mask timestep #3 and #5 because you lack data for
    these timesteps. You can:
        - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:
    ```python
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(LSTM(32))
    ```
    """

    def __init__(self, mask_value=0., data_format='NSXYF', **kwargs):
        super(SequenceMasking, self).__init__(**kwargs)
        self.data_format = data_format
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_mask(self, inputs, mask=None):
        mask = tf.not_equal(inputs, self.mask_value)
        for i, d in enumerate(self.data_format):
            if d not in 'NS':
                mask = tf.keras.backend.any(mask, axis=i, keepdims=True)
        return mask

    def call(self, inputs, **kwargs):
        mask = self.compute_mask(inputs)
        return inputs * tf.cast(mask, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'mask_value': self.mask_value,
                  'data_format': self.data_format}
        base_config = super(SequenceMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskedTimeDistributed(tf.keras.layers.TimeDistributed):
    
    def compute_mask(self, inputs, mask=None):
        return mask
