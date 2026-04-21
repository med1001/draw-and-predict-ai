import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import time
import tensorflow as tf

# Load TFLite model
tflite_model_path = "mnist_cnn_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tkinter setup
WIDTH, HEIGHT = 280, 280  # drawing canvas size
window = tk.Tk()
window.title("Draw a digit (0-9)")

canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg="white")
canvas.grid(row=0, column=0, columnspan=4)

# PIL image to capture drawing
image = Image.new("L", (WIDTH, HEIGHT), color=255)
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def xy(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def add_line(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, x, y, width=15, fill="black", capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill=0, width=15)
    last_x, last_y = x, y

canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", add_line)

def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", (WIDTH, HEIGHT), color=255)
    draw = ImageDraw.Draw(image)

def predict_digit():
    # Resize, invert, blur, normalize
    img = image.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.GaussianBlur(1))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], arr)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred = np.argmax(output)
    
    print(f"Prediction: {pred}")
    print(f"Confidence: {output[pred]:.4f}")
    print(f"Inference time: {(end-start)*1000:.2f} ms")

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.grid(row=1, column=0, pady=5)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=1, pady=5)

window.mainloop()
