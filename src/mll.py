# import system libs
import os
from pathlib import Path
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import random
from PIL import Image

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from keras.applications import VGG16
import shutil

# ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# Create directories for training and testing datasets
train_dir = 'asset/working/train/'
test_dir = 'asset/working/test/'

MODEL_NAME = 'efficientnetb0'
TRANSFER_MODEL_NAME = 'vgg16'
EFFICIENTNETB0_PATH = 'result/efficientnetb0_final_model_new.keras'
VGG16_PATH = 'result/vgg16_final_model.keras'


def fill_model():
    try:
        # Load the model
        loaded_vgg16_model = load_model(VGG16_PATH)

        # Print a summary to verify the loaded model
        loaded_vgg16_model.summary()

        loaded_vgg16_model.trainable = True
        for layer in loaded_vgg16_model.layers:
            layer.trainable = True

        # Now you can use the loaded_vgg16_model for predictions, further training, etc.
        # Example:
        # predictions = loaded_vgg16_model.predict(some_input_data)

        return loaded_vgg16_model

    except OSError as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the file '{VGG16_PATH}' exists and is a valid Keras model file.")
    except ValueError as e: # Handle potential issues with custom objects
        print(f"ValueError loading model: {e}")
        print("This might be due to custom layers or functions. If so, you'll need to provide them in a 'custom_objects' dictionary.")
        # Example for custom objects (if applicable)
        # from your_module import CustomLayer
        # loaded_vgg16_model = load_model(model_path, custom_objects={'CustomLayer': CustomLayer})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def prepare_dataset(data_dir):
    image_paths = list(data_dir.rglob('*.*'))
    image_paths = [str(p) for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']] #filter by extension
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    classes = set(labels)

    # Check the total number of images
    print("="*10)
    print(f'Total number of images: {len(image_paths)}')
    print(f'classes: {classes}')

    # Proceed only if data exists
    if len(image_paths) == 0:
        raise ValueError("No data found, please check the file paths.")

    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    print('Sample of Data')
    print(df.head())

    # Split data into training and testing sets (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Check the sizes of the splits
    print(f'Train set size: {len(train_df)}')
    print(f'Test set size: {len(test_df)}')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move training data to the respective folder
    for _, row in train_df.iterrows():
        label_dir = os.path.join(train_dir, row['label'])
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(row['image_path'], label_dir)

    # Move testing data to the respective folder
    for _, row in test_df.iterrows():
        label_dir = os.path.join(test_dir, row['label'])
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(row['image_path'], label_dir)

    print("Data successfully split into training and testing sets.")

    return classes, train_df, test_df


def get_train_and_val():
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

    return train_generator, val_generator


def get_model_EfficientNetB0(image_shape, train_generator, val_generator):
    efficientnet_model1 = Sequential()

    # Add EfficientNetB0 to the Sequential model and specify the input shape
    efficientnet_model1.add(EfficientNetB0(include_top=False,
                                        input_shape=image_shape,
                                        pooling='avg',
                                        weights='imagenet'))

    efficientnet_model1.add(Flatten())
    efficientnet_model1.add(Dense(256, activation='relu'))
    efficientnet_model1.add(Dropout(0.3))
    efficientnet_model1.add(Dense(128, activation='relu'))
    efficientnet_model1.add(Dropout(0.3))
    efficientnet_model1.add(Dense(64, activation='relu'))
    efficientnet_model1.add(Dropout(0.2))
    efficientnet_model1.add(Dense(5, activation='softmax'))  # Output layer for 5 classes

    # Build the model with a dummy input
    efficientnet_model1.build(input_shape=(None, 224, 224, 3))

    # Compile the model
    efficientnet_model1.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])

    # Display the model summary
    efficientnet_model1.summary()


    # Build the EfficientNetB0 model
    efficientnet_model1 = Sequential()
    efficientnet_model1.add(EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet'))
    efficientnet_model1.add(Flatten())
    efficientnet_model1.add(Dense(128, activation='relu'))
    efficientnet_model1.add(Dropout(0.2))
    efficientnet_model1.add(Dense(64, activation='relu'))
    efficientnet_model1.add(Dense(5, activation='softmax'))  # Output layer for 5 classes

    # Freeze layers
    for layer in efficientnet_model1.layers[:-10]:
        layer.trainable = False

    # Compile the model
    efficientnet_model1.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('efficientnetb0_best_updated.keras', save_best_only=True)

    # Train the model
    history = efficientnet_model1.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[early_stopping, model_checkpoint])

    # Visualize training results
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    save_plot_result(plt, MODEL_NAME + '_accuracy.jpg')

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    save_plot_result(plt, MODEL_NAME + '_loss.jpg')

    # Generate predictions
    predictions = efficientnet_model1.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)  # Predicted classes
    y_true = val_generator.classes           # True classes

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

    # Metrics such as Accuracy, Precision, Recall, and F1-Score
    test_loss, test_accuracy = efficientnet_model1.evaluate(val_generator)
    print(f"\nValidation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")

    # Get predictions
    predictions = efficientnet_model1.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)  # Predicted classes
    y_true = val_generator.classes           # True classes

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices.keys())

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d')
    plt.title("Confusion Matrix for EfficientNetB0")
    save_plot_result(plt, MODEL_NAME + '_confusion_matrix.jpg')

    efficientnet_model1.save(EFFICIENTNETB0_PATH)


def transfer_learning_VGG16(image_shape, class_counts, train_generator, val_generator):
    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape, pooling='max')

    # Freeze the first 15 layers
    for layer in base_model.layers[:15]:
        layer.trainable = False

    # Customize the top layers (fine-tuning)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)  # Dropout to prevent overfitting
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(class_counts, activation='softmax')(x)

    # Build the model
    VGG16_model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    VGG16_model.compile(optimizer=Adamax(learning_rate=0.0001), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

    # Display the model summary
    VGG16_model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('vgg16_fine_tuned_best.keras', save_best_only=True)

    # Train the model
    epochs = 20
    VGG16_history = VGG16_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=False,
        verbose=1
    )

    # Visualize training results for VGG16
    plt.plot(VGG16_history.history['accuracy'], label='Training Accuracy')
    plt.plot(VGG16_history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('VGG16 Model Accuracy')
    save_plot_result(plt, TRANSFER_MODEL_NAME + '_accuracy.jpg')

    plt.plot(VGG16_history.history['loss'], label='Training Loss')
    plt.plot(VGG16_history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('VGG16 Model Loss')
    save_plot_result(plt, TRANSFER_MODEL_NAME + '_loss.jpg')

    # Generate predictions for VGG16
    predictions_vgg = VGG16_model.predict(val_generator)
    y_pred_vgg = np.argmax(predictions_vgg, axis=1)
    y_true_vgg = val_generator.classes

    # Classification Report for VGG16
    print("VGG16 Classification Report:")
    print(classification_report(y_true_vgg, y_pred_vgg, target_names=list(val_generator.class_indices.keys())))

    # Metrics for VGG16
    test_loss_vgg, test_accuracy_vgg = VGG16_model.evaluate(val_generator)
    print(f"\nValidation Loss (VGG16): {test_loss_vgg:.4f}, Validation Accuracy (VGG16): {test_accuracy_vgg:.4f}")

    # Confusion Matrix for VGG16
    cm_vgg = confusion_matrix(y_true_vgg, y_pred_vgg)
    disp_vgg = ConfusionMatrixDisplay(confusion_matrix=cm_vgg, display_labels=val_generator.class_indices.keys())

    # Visualize the confusion matrix for VGG16
    plt.figure(figsize=(10, 8))
    disp_vgg.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d')
    plt.title("Confusion Matrix for VGG16")
    save_plot_result(plt, TRANSFER_MODEL_NAME + '_confusion_matrix.jpg')

    VGG16_model.save(VGG16_PATH)


def draw_rectangle_on_detection(img_path, model, last_conv_layer_name, class_indices):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Adjust based on the model

    # Get model predictions
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds[0])
    pred_class_name = list(class_indices.keys())[list(class_indices.values()).index(pred_class_idx)]
    print(f"Predicted Class: {pred_class_name} (Index: {pred_class_idx})")

    # Get the last convolutional layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(inputs=model.input, outputs=last_conv_layer.output)

    # Grad-CAM: Get gradients and conv layer output
    with tf.GradientTape() as tape:
        conv_output = last_conv_layer_model(img_array)
        tape.watch(conv_output)
        preds = model(img_array)
        loss = preds[:, pred_class_idx]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        raise RuntimeError("Error: Gradients could not be computed.")

    # Compute pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]  # Remove batch dimension

    # Multiply pooled gradients with feature maps
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1).numpy()

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load the original image
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + original_img

    # Detect the bounding box from the heatmap
    thresh = np.where(heatmap[:, :, 0] > heatmap[:, :, 0].mean(), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("Warning: No contours found in heatmap.")

    # Save the output image
    current_directory_path = os.getcwd()
    output_filename = os.path.join(current_directory_path, 'result', f'{pred_class_name}_detection.jpg')
    cv2.imwrite(output_filename, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {output_filename}")

    return superimposed_img


def save_plot_result(plt_ref, file_name):
    current_directory_path = os.getcwd()
    directory_path = os.path.join(current_directory_path, 'result')
    
    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    file_path = os.path.join(directory_path, file_name)
    plt_ref.savefig(file_path)


def plot_average_height_width(classes, train_dir):
    class_averages = {}

    for cls in classes:
        class_dir = os.path.join(train_dir, cls)
        heights = []
        widths = []
        
        for image_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image_name)
            img = mpimg.imread(img_path)
            d1, d2, _ = img.shape
            heights.append(d1)
            widths.append(d2)
        
        # Calculate the average height and width
        avg_height = np.mean(heights)
        avg_width = np.mean(widths)
        class_averages[cls] = (avg_height, avg_width)

    # Print the results
    for cls, (avg_height, avg_width) in class_averages.items():
        print(f'{cls}: Average Height = {avg_height:.2f}, Average Width = {avg_width:.2f}')

    # Create a DataFrame for average dimensions per class
    class_averages_df = pd.DataFrame.from_dict(class_averages, orient='index', columns=['Average Height', 'Average Width'])

    # Plot the average dimensions
    class_averages_df.plot(kind='bar', figsize=(10, 6))
    title = "average_image_dimensions_per_class"
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Pixels")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    save_plot_result(plt, title + '.jpg')


def plot_distribution(dataframe):
    sns.countplot(data=dataframe, x='label')
    plt.title("Class Distribution in Training Data")
    plt.xticks(rotation=45)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    save_plot_result(plt, 'distribution.jpg')


def plot_random_images(classes, dataframe):
    plt.figure(figsize=(15, 10))

    # Show a random image from each class
    for i, cls in enumerate(classes):
        class_dir = os.path.join(dataframe, cls)
        sample_image = random.choice(os.listdir(class_dir))
        img_path = os.path.join(class_dir, sample_image)
        img = Image.open(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')

    plt.tight_layout()
    save_plot_result(plt, 'random_image.jpg')


def plot_distribution_height_width(classes, dataframe):
    heights = []
    widths = []

    for cls in classes:
        class_dir = os.path.join(dataframe, cls)
        for image_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image_name)
            img = mpimg.imread(img_path)
            d1, d2, _ = img.shape
            heights.append(d1)
            widths.append(d2)

    # Plot height and width distributions
    plt.figure(figsize=(12, 6))
    plt.hist(heights, bins=30, alpha=0.7, label='Height')
    plt.hist(widths, bins=30, alpha=0.7, label='Width')
    title = "distribution_of_image_heights_and_widths"
    plt.title(title)
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.legend()
    save_plot_result(plt, title + '.jpg')

    avg_height = np.mean(heights)
    avg_width = np.mean(widths)
    print(f"Average Image Dimensions: {avg_height:.2f}x{avg_width:.2f}")

    sample_image_path = random.choice(dataframe['image_path'])
    img = load_img(sample_image_path, target_size=(int(avg_height), int(avg_width)))
    plt.imshow(img)
    title = "resized_example_image"
    plt.title(title)
    plt.axis('off')
    save_plot_result(plt, title + '.jpg')


def plot_rgb_histogram(classes, dataframe):
    # Plot RGB histogram for a random image from each class
    plt.figure(figsize=(15, 10))

    for i, cls in enumerate(classes):
        class_dir = os.path.join(dataframe, cls)
        sample_image = random.choice(os.listdir(class_dir))
        img_path = os.path.join(class_dir, sample_image)
        
        # Load and convert image to numpy array
        img = Image.open(img_path)
        img_array = np.array(img)

        # Plot RGB histogram
        plt.subplot(2, 3, i + 1)
        for j, color in enumerate(['red', 'green', 'blue']):
            plt.hist(img_array[..., j].ravel(), bins=256, color=color, alpha=0.5)
        plt.title(f"RGB Histogram for {cls}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(False)

    plt.tight_layout()
    save_plot_result(plt, 'rgb_histogram.jpg')


def pre_procesing():
    # Data path
    data_dir = Path('asset/flower/train')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    classes, train_df, test_df = prepare_dataset(data_dir)
    
    # plot_average_height_width(classes, train_dir)
    # plot_distribution(train_df)
    # plot_random_images(classes, train_df)
    # plot_distribution_height_width(classes, train_df)
    # plot_rgb_histogram(classes, train_df)

    # Define the image dimensions
    image_shape = (224, 224, 3)
    class_counts = 5
    train_generator, val_generator = get_train_and_val()
    get_model_EfficientNetB0(image_shape, train_generator, val_generator)
    transfer_learning_VGG16(image_shape, class_counts, train_generator, val_generator)


def test_model():
    VGG16_model = fill_model()
    img_path = "asset/flower/test/na/Image_1.jpg"
    last_conv_layer_name = VGG16_model.layers[-1].name
    class_indices = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    # Get the predicted class name
    result_img = draw_rectangle_on_detection(img_path, VGG16_model, last_conv_layer_name, class_indices)
    if result_img is None:
        print("Error processing image.")


if __name__ == "__main__":
    test_model()