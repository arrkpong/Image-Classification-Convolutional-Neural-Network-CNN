import tensorflow as tf
import os
import subprocess
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import socket
import time
from tqdm import tqdm
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ImageClassifierApp:
    def __init__(self):
        st.set_page_config(page_title="Image Classification App", layout="wide")
        self.init_environment()
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.history = None
        self.model = self.load_trained_model()
        self.tensorboard_port = self.get_free_port()
        self.app()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def init_environment(self):
        """Initialize environment settings."""
        st.sidebar.title("Configuration")
        self.use_onednn = st.sidebar.checkbox("Enable ONEDNN Optimizations (TF_ENABLE_ONEDNN_OPTS=1)", value=True)
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' if self.use_onednn else '0'
        message = "ONEDNN optimizations are enabled." if self.use_onednn else "ONEDNN optimizations are disabled."
        st.sidebar.write(f"**Status:** {message}")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_data(self):
        """Load and preprocess CIFAR-10 dataset with caching."""
        with st.spinner("Loading dataset..."):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
        st.sidebar.success("Dataset loaded successfully!")
        return x_train, y_train, x_test, y_test
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_free_port(self):
        """Find a free port for TensorBoard."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
        sock.close()
        return port
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_trained_model(self):
        """Load a pre-trained model if available, otherwise build a new model."""
        if os.path.exists('model_checkpoint.keras'):
            st.sidebar.info("Loading pre-trained model...")
            return tf.keras.models.load_model('model_checkpoint.keras')
        else:
            st.sidebar.info("No pre-trained model found. Building and training a new model.")
            return self.build_model()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def build_model(self):
        """Build the CNN model."""
        model = tf.keras.models.Sequential([ 
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer='adam',  # Default optimizer if none provided
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def train_model(self, optimizer, epochs, batch_size):
        """Train the model with dynamic learning rate scheduler."""
        try:
            checkpoint_filepath = 'model_checkpoint.keras'
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15)
            ]
            
            # Check if optimizer is not None
            if optimizer is None:
                st.error("Optimizer is not defined!")
                return
            
            st.sidebar.info("Training model...")
            self.history = self.model.fit(
                self.x_train, self.y_train,
                epochs=epochs,
                validation_data=(self.x_test, self.y_test),
                callbacks=callbacks,
                batch_size=batch_size
            )
            pd.DataFrame(self.history.history).to_csv('training_history.csv', index=False)
            st.sidebar.success("Model training complete! History saved to 'training_history.csv'.")
            
            # Plot training history (accuracy and loss)
            self.plot_training_history()

        except Exception as e:
            st.error(f"Training error: {e}")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def plot_training_history(self):
        """Plot training accuracy and loss."""
        if self.history is not None:
            accuracy = self.history.history['accuracy']
            val_accuracy = self.history.history['val_accuracy']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            # Create DataFrame for easy plotting
            df = pd.DataFrame({
                'Epochs': range(1, len(accuracy) + 1),
                'Training Accuracy': accuracy,
                'Validation Accuracy': val_accuracy,
                'Training Loss': loss,
                'Validation Loss': val_loss
            })

            st.subheader("Training Progress")
            st.line_chart(df[['Training Accuracy', 'Validation Accuracy']], use_container_width=True)
            st.line_chart(df[['Training Loss', 'Validation Loss']], use_container_width=True)
            st.write(f"Highest Training Accuracy: {round(max(self.history.history['accuracy']), 4)}")
            st.write(f"Highest Testing Accuracy: {round(max(self.history.history['val_accuracy']), 4)}")
            st.write(f"Lowest Training Loss: {round(min(self.history.history['loss']), 4)}")
            st.write(f"Lowest Testing Loss: {round(min(self.history.history['val_loss']), 4)}")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def display_tensorboard(self):
        """Start TensorBoard and display link."""
        tensorboard_url = f"http://localhost:{self.tensorboard_port}"
        subprocess.Popen(['tensorboard', '--logdir', './logs', '--host', 'localhost', '--port', str(self.tensorboard_port)])
        st.markdown(f"Access TensorBoard: [Open TensorBoard]({tensorboard_url})")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def app(self):
        """Run the Streamlit application."""
        st.title("Image Classification App")
        st.info("Predict CIFAR-10 images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.")

        uploaded_files = st.file_uploader("Upload images (JPG only):", type="jpg", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file).resize((32, 32))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    img_array = tf.expand_dims(img_array, 0)
                    predictions = self.model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    class_name = self.class_names[np.argmax(score)]
        
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                    st.write(f"**Predicted Class:** {class_name}")
                    st.write(f"**Confidence:** {round(100 * np.max(score), 2)}%")
                except Exception as e:
                    st.error(f"Error processing image: {e}")

        st.sidebar.header("Training Configuration")
        
        # Learning Rate Input
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=0.001, format="%.6f")
        
        # Epoch Input
        epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=20)

        # Batch Size Input
        batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)

        # Optimizer Selection
        optimizer_options = {
            "Adam": tf.keras.optimizers.Adam(learning_rate),
            "SGD": tf.keras.optimizers.SGD(learning_rate),
            "RMSprop": tf.keras.optimizers.RMSprop(learning_rate),
            "Adagrad": tf.keras.optimizers.Adagrad(learning_rate),
            "Adadelta": tf.keras.optimizers.Adadelta(learning_rate),
            "FTRL": tf.keras.optimizers.Ftrl(learning_rate),
            "Nadam": tf.keras.optimizers.Nadam(learning_rate)
        }
        selected_optimizer = st.sidebar.selectbox("Optimizer", list(optimizer_options.keys()))
        selected_optimizer = optimizer_options[selected_optimizer]

        if st.sidebar.button("Train Model"):
            self.train_model(selected_optimizer, epochs, batch_size)

        if st.sidebar.button("Start TensorBoard"):
            self.display_tensorboard()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ImageClassifierApp()
