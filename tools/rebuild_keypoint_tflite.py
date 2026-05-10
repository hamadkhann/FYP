import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
NUM_CLASSES = 26
DATASET = "model/keypoint_classifier/keypoint.csv"
KERAS_PATH = "model/keypoint_classifier/keypoint_classifier.keras"
TFLITE_PATH = "model/keypoint_classifier/keypoint_classifier.tflite"


def main() -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    x_dataset = np.loadtxt(
        DATASET, delimiter=",", dtype="float32", usecols=list(range(1, (21 * 2) + 1))
    )
    y_dataset = np.loadtxt(DATASET, delimiter=",", dtype="int32", usecols=(0))

    x_train, x_test, y_train, y_test = train_test_split(
        x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((21 * 2,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                128, activation="mish", kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                64, activation="mish", kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                32, activation="mish", kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Keep training short enough for local rebuild while producing a usable model.
    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    model.save(KERAS_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_quantized_model)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Rebuilt model. val_loss={loss:.4f}, val_acc={acc:.4f}")
    print(f"Wrote: {KERAS_PATH}")
    print(f"Wrote: {TFLITE_PATH} ({len(tflite_quantized_model)} bytes)")


if __name__ == "__main__":
    main()
