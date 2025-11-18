import tensorflow as tf
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, att_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.att_dim = att_dim

    def build(self, input_shape):
        self.W_q = self.add_weight(
            shape=(input_shape[-1], self.att_dim),
            initializer="glorot_uniform",
            name="W_q"
        )
        self.W_k = self.add_weight(
            shape=(input_shape[-1], self.att_dim),
            initializer="glorot_uniform",
            name="W_k"
        )
        self.W_v = self.add_weight(
            shape=(input_shape[-1], self.att_dim),
            initializer="glorot_uniform",
            name="W_v"
        )

    def call(self, inputs, mask=None):
        Q = inputs @ self.W_q
        K = inputs @ self.W_k
        V = inputs @ self.W_v

        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.att_dim, tf.float32)
        )

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=1)
            scores += (1.0 - mask) * -1e9

        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"att_dim": self.att_dim})
        return config
