import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "plant_disease_model.keras"
)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters:", model.count_params())