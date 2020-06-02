import tensorflow as tf

def build_model(input_shape):
    if isinstance(input_shape, int):
        input_shape = [input_shape]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse']
        )
    return model

if __name__ == '__main__':
    print(build_model(10).summary())