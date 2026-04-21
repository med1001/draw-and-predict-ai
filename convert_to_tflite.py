import tensorflow as tf

# Load saved model
converter = tf.lite.TFLiteConverter.from_saved_model("mnist_cnn_model")

# Enable optimization (basic quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save TFLite model
with open("mnist_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'mnist_cnn_model.tflite'")
