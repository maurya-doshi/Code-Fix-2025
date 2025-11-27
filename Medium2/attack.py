import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

MODEL_PATH = "fashion_classifier.h5"
IMAGES_DIR = "images"
FLAG_PATH = "Flag_image.png"
EPSILON = 0.03   # FGSM attack step size

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📂 Searching for images...")
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")]

print("🖼️ Found images:", image_files)

if not image_files:
    raise Exception("No PNG files found in images directory.")

# ==============================
# LOOP THROUGH ALL IMAGES
# ==============================
for img_file in image_files:

    print(f"\n🎯 Attacking image: {img_file}")
    path = os.path.join(IMAGES_DIR, img_file)

    # Load + preprocess
    img = load_img(path, target_size=(28, 28), color_mode="grayscale")
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    x = tf.convert_to_tensor(img, dtype=tf.float32)

    # ---- FGSM ----
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        label = tf.argmax(pred, axis=1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, pred)

    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    adv = tf.clip_by_value(x + EPSILON * signed_grad, 0, 1)

    # ---- Predictions ----
    orig_pred = model.predict(x, verbose=0)
    adv_pred = model.predict(adv, verbose=0)

    orig_class = np.argmax(orig_pred)
    adv_class = np.argmax(adv_pred)

    # Print info
    print(f"  Original: {orig_class} | conf: {float(np.max(orig_pred)):.3f}")
    print(f"  Adversarial: {adv_class} | conf: {float(np.max(adv_pred)):.3f}")

    # ---------------------
    # SUCCESS? SAVE FLAG
    # ---------------------
    if adv_class != orig_class:
        print("🎉 SUCCESS — model fooled on this image!")

        adv_pixels = (adv[0] * 255).numpy().astype("uint8").reshape(28, 28)
        Image.fromarray(adv_pixels).save(FLAG_PATH)

        print("🏁 Saved flag:", FLAG_PATH)
        print("➡ Attack completed.")
        break

else:
    print("\n❌ No image could fool the model. Increase epsilon or try again.")
