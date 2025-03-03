
import tensorflow as tf
import tensorflow_transform as tft
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
import os

# Model function
def _build_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function untuk parsing TFRecord berdasarkan schema
def _parse_function(serialized_example, schema):
    # Konversi schema menjadi feature_spec
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    parsed_example = tf.io.parse_single_example(serialized_example, feature_spec)
    return parsed_example

def run_fn(fn_args):
    schema_path = "/Users/bernadetta/Desktop/ProyekMLOps1/bernadettasri-pipeline/SchemaGen/schema/6/schema.pb"
    schema = schema_pb2.Schema()

    # Pastikan path schema benar
    if not tf.io.gfile.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Membaca schema
    with tf.io.gfile.GFile(schema_path, "rb") as f:
        schema.ParseFromString(f.read())
    print("Schema parsed successfully.")

    # Ambil path dataset dari Transform
    train_dataset_path = fn_args.train_files[0]  
    eval_dataset_path = fn_args.eval_files[0]

    print("Train dataset path:", train_dataset_path)
    print("Eval dataset path:", eval_dataset_path)

    if not tf.io.gfile.exists(train_dataset_path) or not tf.io.gfile.exists(eval_dataset_path):
        raise FileNotFoundError("Train or Eval dataset files not found!")

    # Parsing dataset menggunakan schema
    train_dataset = (tf.data.TFRecordDataset(train_dataset_path, compression_type="GZIP")
                     .map(lambda x: _parse_function(x, schema))
                     .batch(32)
                     .shuffle(1000))
    
    eval_dataset = (tf.data.TFRecordDataset(eval_dataset_path, compression_type="GZIP")
                    .map(lambda x: _parse_function(x, schema))
                    .batch(32))

    # Bangun model
    model = _build_keras_model()

    # Training
    model.fit(train_dataset, validation_data=eval_dataset, epochs=5)

    # Simpan model
    model.save(fn_args.serving_model_dir, save_format='tf')

