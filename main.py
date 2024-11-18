import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ImageClassifierApp:
    def __init__(self):
        # Load and preprocess the dataset
        print("Loading and preprocessing the dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Build the CNN model
        self.model = self.build_model()

        # Variable to store training history
        self.history = None

        # Class names for CIFAR-10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # Run the Streamlit application
        self.app()
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
            # Define callbacks
            checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True)
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
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            # Preprocess the uploaded image
            print("Preprocessing the uploaded image...")
            img = Image.open(uploaded_file)
            img = img.resize((32, 32))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = tf.expand_dims(img_array, 0)

            # Display training status
            st.text("Training the model... Please wait.")

            # Train the model
            self.train_model()

            # Make predictions using the trained model
            if self.history is not None and 'accuracy' in self.history.history:
                predictions = self.model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                class_name = self.class_names[np.argmax(score)]

                # Display the prediction result
                print("Prediction completed!")
                st.success("Prediction completed!")
                st.image(img, use_column_width=True)
                st.write(f"Predicted class: {class_name}")
                st.write(f"Confidence: {round(100 * np.max(score), 2)}%")  
                
                # Display training history charts
                st.title("Model Training History")
                st.write("Charts showing the accuracy and loss of the model during training")
                st.line_chart({'Training Accuracy': self.history.history['accuracy'],
                               'Testing Accuracy': self.history.history['val_accuracy']})
                st.line_chart({'Training Loss': self.history.history['loss'],
                               'Testing Loss': self.history.history['val_loss']})
                st.write(f"Highest Training Accuracy: {round(max(self.history.history['accuracy']), 4)}")
                st.write(f"Highest Testing Accuracy: {round(max(self.history.history['val_accuracy']), 4)}")
                st.write(f"Lowest Training Loss: {round(min(self.history.history['loss']), 4)}")
                st.write(f"Lowest Testing Loss: {round(min(self.history.history['val_loss']), 4)}")

            else:
                # Display an error message if training fails
                st.error("Model training failed. Please check the error message above.")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = ImageClassifierApp()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------