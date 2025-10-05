import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset(data_dir, train_dir, test_dir):
    image_paths = list(data_dir.rglob("*.*"))
    image_paths = [
        str(p)
        for p in image_paths
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    ]  # filter by extension
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    classes = set(labels)

    # Check the total number of images
    print("=" * 10)
    print(f"Total number of images: {len(image_paths)}")
    print(f"classes: {classes}")

    # Proceed only if data exists
    if len(image_paths) == 0:
        raise ValueError("No data found, please check the file paths.")

    # Create a DataFrame
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    print("Sample of Data")
    print(df.head())

    # Split data into training and testing sets (80% train, 20% test)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Check the sizes of the splits
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move training data to the respective folder
    for _, row in train_df.iterrows():
        label_dir = os.path.join(train_dir, row["label"])
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(row["image_path"], label_dir)

    # Move testing data to the respective folder
    for _, row in test_df.iterrows():
        label_dir = os.path.join(test_dir, row["label"])
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(row["image_path"], label_dir)

    print("Data successfully split into training and testing sets.")

    return classes, train_df, test_df
    

def save_plot_result(plt_ref, file_name):
    current_directory_path = os.getcwd()
    directory_path = os.path.join(current_directory_path, "result")

    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    file_path = os.path.join(directory_path, file_name)
    plt_ref.savefig(file_path)
