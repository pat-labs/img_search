import os
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from pydantic import BaseModel

from algorithms_keypoints_descriptors import execSift, readImage

train_data_path = "asset/flower/train"


class ImageData(BaseModel):
    filename: str
    label: int | None
    descriptors: list


def loadImages(root_folder: str) -> Tuple[List[ImageData], List[str]]:
    images = list()
    labels = set()

    for current_folder, subfolders, files in os.walk(root_folder):
        folder_label = os.path.basename(current_folder)
        if files:
            labels.add(folder_label)
    sorted_labels = sorted(labels)

    for current_folder, subfolders, files in os.walk(root_folder):
        folder_label = os.path.basename(current_folder)
        if files:
            label_index = sorted_labels.index(folder_label)
            for filename in files:
                file_path = os.path.join(current_folder, filename)
                image_matrix = readImage(file_path, 80, 80)
                if image_matrix is not None:
                    _, des = execSift(image_matrix)
                    # if isinstance(des, np.ndarray):
                    #     des = des.tolist()
                    image = ImageData(
                        filename=filename, label=label_index, descriptors=des
                    )
                    images.append(image)

    return images, sorted_labels


def buildIndex(vectors: np.array) -> faiss.Index:
    vectors = vectors.astype("float32")
    vector_dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(vector_dimension)
    # index = faiss.IndexFlatL2(vector_dimension)

    # quantizer = faiss.IndexFlatL2(vector_dimension)
    # index = faiss.IndexIVFFlat(quantizer, vector_dimension, 100)
    # index = faiss.IndexHNSWFlat(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    return index


def saveIndex(index, filename):
    faiss.write_index(index, filename)


def loadIndex(filename):
    return faiss.read_index(filename)


def findSimilarImages(
    index: faiss.Index, query_image_path: str, k: int
) -> pd.DataFrame:
    image_matrix = readImage(query_image_path, 80, 80)
    _, search_descriptor = execSift(image_matrix)
    vector = np.vstack(search_descriptor)
    faiss.normalize_L2(search_descriptor)

    distances, ann = index.search(vector, k=k)
    results = pd.DataFrame({"distances": distances[0], "ann": ann[0]})

    return results


def main() -> None:
    train_images, train_labels = loadImages(train_data_path)
    descriptors = np.vstack(
        [item.descriptors for item in train_images if item.descriptors is not None]
    )
    index = buildIndex(descriptors)

    query_image_path = "asset/flower/test/na/Image_1.jpg"
    k = index.ntotal
    print(k)
    results = findSimilarImages(index, query_image_path, k)
    df_images = pd.DataFrame([item.__dict__ for item in train_images])

    merged_results = results.merge(df_images, left_on="ann", right_index=True)
    sorted_results = merged_results.sort_values("distances")
    print(sorted_results)


if __name__ == "__main__":
    main()
