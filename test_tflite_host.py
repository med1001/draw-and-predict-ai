import numpy as np
import tensorflow as tf
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mnist_cnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load real MNIST test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # (N, 28, 28, 1)

# Keep it fast for a quick host-side check
NUM_SAMPLES = 1000
x_eval = x_test[:NUM_SAMPLES]
y_eval = y_test[:NUM_SAMPLES]

# Optional warm-up run
interpreter.set_tensor(input_details[0]['index'], x_eval[0:1])
interpreter.invoke()

correct = 0
total_inference_ms = 0.0

for i in range(NUM_SAMPLES):
    input_data = x_eval[i:i + 1]
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end = time.perf_counter()

    pred = int(np.argmax(output))
    if pred == int(y_eval[i]):
        correct += 1
    total_inference_ms += (end - start) * 1000.0

accuracy = correct / NUM_SAMPLES
avg_inference_ms = total_inference_ms / NUM_SAMPLES

print(f"Samples tested: {NUM_SAMPLES}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Average inference time: {avg_inference_ms:.3f} ms/sample")
