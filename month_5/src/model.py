import tensorflow as tf

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),  # ✅ correct way
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model