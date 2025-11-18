import tensorflow as tf
import numpy as np
from custom_layers import ScaledDotProductAttention

MODEL_PATH = "bilstm_att_correct_mask_final.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"ScaledDotProductAttention": ScaledDotProductAttention},
    compile=False
)

print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== OUTPUT SHAPE =====")
print(model.output_shape)

# Tạo input giả để kiểm tra số lớp
input_shape = model.input_shape[1]
dummy = np.zeros((1, input_shape), dtype=np.int32)

pred = model.predict(dummy)[0]
print("\n===== RAW PREDICTION =====")
print(pred)
print("Prediction length:", len(pred))
