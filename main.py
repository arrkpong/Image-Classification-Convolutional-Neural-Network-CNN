import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os  # Import os module for checking and creating files/folders

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ImageClassifierApp:
    def __init__(self):
        # Load and preprocess the dataset
        print("Loading and preprocessing the dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Build the CNN model
        self.model = self.load_trained_model()  # Load trained model if available

        # Variable to store training history
        self.history = None

        # Class names for CIFAR-10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # Run the Streamlit application
        self.app()

    def load_trained_model(self):
        if os.path.exists('model_checkpoint.keras'):
            print("Loading the pre-trained model...")
            return tf.keras.models.load_model('model_checkpoint.keras')
        else:
            print("Building a new model...")
            return self.build_model()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def build_model(self):
        # Define the architecture of the Convolutional Neural Network (CNN) model
        print("Building the CNN model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.5),
            layers.Dense(10)
        ])

        # Compile the model
        print("Compiling the model...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def train_model(self):
        try:
            # Check if the 'logs' directory exists, if not, create it
            if not os.path.exists('./logs'):
                os.makedirs('./logs')

            # Check if the checkpoint file exists, if not, create it
            checkpoint_filepath = 'model_checkpoint.keras'
            checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True)
            
            # Define other callbacks
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15)
            tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
            
            def lr_schedule(epoch):
                return 0.001 * 0.9 ** epoch
            lr_scheduler_callback = LearningRateScheduler(lr_schedule)
            
            reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15)

            # Train the model with callbacks
            print("Training the model...")
            self.history = self.model.fit(
                self.x_train, self.y_train, epochs=20, validation_data=(self.x_test, self.y_test),
                callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback, lr_scheduler_callback, reduce_lr_callback]
            )
            
            # Save training history to CSV
            df = pd.DataFrame(self.history.history)
            df.to_csv('training_history.csv', index=False)
            print("Training history saved to 'training_history.csv'.")

        except Exception as e:
            st.error(f"An error occurred during training: {e}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def app(self):
        # Streamlit application layout
        st.title("Image Classification App")
        st.info("Note: The program can predict images in the categories available in the CIFAR-10 dataset: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.")
        
        uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

        if uploaded_files:
            # Preprocess and make predictions for each uploaded file
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                img = img.resize((32, 32))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                img_array = tf.expand_dims(img_array, 0)

                # Display training status
                with st.spinner('Training the model... Please wait.'):
                    self.train_model()

                # Make predictions using the trained model
                if self.history is not None and 'accuracy' in self.history.history:
                    predictions = self.model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    class_name = self.class_names[np.argmax(score)]

                    # Display the prediction result
                    st.success("Prediction completed!")
                    st.image(img, use_column_width=True)
                    st.write(f"Predicted class: {class_name}")
                    st.write(f"Confidence: {round(100 * np.max(score), 2)}%")  

                else:
                    st.error("Model training failed. Please check the error message above.")
        
        # Display TensorBoard (if available)
        st.title("TensorBoard")
        tensorboard_log_dir = './logs'
        st.tensorboard(tensorboard_log_dir)

        # Display training history charts
        if self.history is not None:
            st.title("Model Training History")
            st.write("Charts showing the accuracy and loss of the model during training")
            st.line_chart({
                'Training Accuracy': self.history.history['accuracy'],
                'Testing Accuracy': self.history.history['val_accuracy']
            })
            st.line_chart({
                'Training Loss': self.history.history['loss'],
                'Testing Loss': self.history.history['val_loss']
            })
            st.write(f"Highest Training Accuracy: {round(max(self.history.history['accuracy']), 4)}")
            st.write(f"Highest Testing Accuracy: {round(max(self.history.history['val_accuracy']), 4)}")
            st.write(f"Lowest Training Loss: {round(min(self.history.history['loss']), 4)}")
            st.write(f"Lowest Testing Loss: {round(min(self.history.history['val_loss']), 4)}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = ImageClassifierApp()
