from collections import defaultdict

import faiss
import numpy as np
from skimage.feature import ORB, fisher_vector
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from utils import load_descriptors, load_gmm_model, load_keypoints


def split_data(data, M, Ks):
    D = data.shape[1]
    assert D % M == 0, "D must be divisible by M"
    sub_vectors = []
    for i in range(M):
        sub_vectors.append(data[:, i * (D // M) : (i + 1) * (D // M)])
    return sub_vectors


def quantize_data(data, codebooks):
    sub_vectors = split_data(data, len(codebooks), [c.n_clusters for c in codebooks])
    codes = []
    for sub_vector, codebook in zip(sub_vectors, codebooks):
        codes.append(codebook.predict(sub_vector))
    return np.vstack(codes).T


def train_codebooks(sub_vectors, Ks):
    codebooks = []
    for sub_vector in sub_vectors:
        kmeans = KMeans(n_clusters=Ks, random_state=0).fit(sub_vector)
        codebooks.append(kmeans)
    return codebooks


def reduce_dimensionality(image_descriptors, n_components=64):
    pca = PCA(n_components=n_components)
    reduced_descriptors = pca.fit_transform(image_descriptors)
    return reduced_descriptors, pca


def create_visual_vocabulary(images_descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(images_descriptors)
    return kmeans.cluster_centers_, kmeans


def create_histogram(image_descriptors, images_vocabulary):
    # Predict the clusters for each descriptor
    codebook_idx = np.argmin(
        np.linalg.norm(image_descriptors[:, np.newaxis, :] - images_vocabulary, axis=2),
        axis=1,
    )

    # Create a histogram of size num_clusters
    hist, _ = np.histogram(
        codebook_idx, bins=len(images_vocabulary), range=(0, len(images_vocabulary))
    )

    # Ensure the histogram has the correct shape
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)

    return hist


def compute_fisher_vectors(image_descriptors, gmm):
    fisher_vectors = []
    for descriptor in image_descriptors:
        fv = fisher_vector(descriptor, gmm)
        fisher_vectors.append(fv)
    return np.array(fisher_vectors)


def fake_images_descriptors(images_descriptors):
    # TODO: For test
    # Check the number of samples
    num_samples = images_descriptors.shape[0]

    # Set n_components to be less than or equal to num_samples
    n_components = min(50, num_samples)

    if num_samples < 2:
        # Handle case with fewer than 2 samples by duplicating
        images_descriptors = np.tile(images_descriptors, (2, 1))

    return images_descriptors


def get_fisher_vector(images_descriptors):
    num_samples = images_descriptors.shape[0]

    # Set n_components to be less than or equal to num_samples
    n_components = min(50, num_samples)

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="diag", random_state=0
    )
    gmm.fit(images_descriptors)
    # Inspect shapes
    print("Descriptor shape:", images_descriptors.shape)
    print("Means shape:", gmm.means_.shape)
    print("Covariances shape:", gmm.covariances_.shape)
    print("Weights shape:", gmm.weights_.shape)

    # Ensure shapes are compatible
    if gmm.covariances_.shape[1:] != images_descriptors.shape[1:]:
        raise ValueError("Shape mismatch between covariances and descriptor features.")

    fv = fisher_vector(images_descriptors, gmm)
    return fv


def combine_features(image_histogram, image_fisher_vector):
    combined_feature = np.concatenate((image_histogram, image_fisher_vector))
    return combined_feature


def build_inverted_index(quantized_codes):
    inverted_index = defaultdict(list)
    for i, code in enumerate(quantized_codes):
        inverted_index[tuple(code)].append(i)
    return inverted_index


def create_faiss_index(features):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] == 0:
        raise ValueError("Features array is empty")
    print("Features shape:", features.shape)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index


def search_similar_images(query_fisher_vector, index, k=5):
    D, I = index.search(query_fisher_vector, k)
    # Filter out invalid indices
    valid_indices = [index for index in I[0] if index != -1]

    if not valid_indices:
        print("No valid similar images found.")
        return []

    return valid_indices


def main():
    """
    IVFADC
    _summary_
    1. Create BoF
       1.1. Extracción de Características SIFT y Almacenamiento (formato pkl)
       1.2. Product Quantization
        * Construcción del vocabulario visual Utilizar un algoritmo de clustering como k-means para agrupar los descriptores SIFT en un número fijo de clusters. Cada cluster representa una "palabra visual".
       1.3. Construcción de Histogramas de Palabras Visuales
    * Para cada imagen, asignar cada descriptor SIFT al cluster más cercano (su "palabra visual").
    * Contar la frecuencia de cada palabra visual en la imagen para construir el histograma.
    2. Cálculo de Fisher Vectors
    * Calcular la matriz de Fisher basada en los histogramas.
    * Utilizar la matriz de Fisher para calcular la similitud entre pares de imágenes.
    3. Creación del Índice Faiss
    4. Búsqueda de Imágenes Similares
    _detail_
    1. Extraer características: Aplicar extract_sift_features a todas las imágenes y guardar los descriptores.
    2. Crear vocabulario visual: Concatenar todos los descriptores y usar create_visual_vocabulary para obtener los centros de los clusters.
    3. Construir histogramas: Para cada imagen, calcular el histograma utilizando create_histogram.
    4. Calcular Fisher Vectors: Utilizar compute_fisher_vectors para obtener los vectores de Fisher.
    5. Crear índice Faiss: Usar create_faiss_index para crear el índice.
    6. Buscar imágenes similares: Para una imagen de consulta, calcular su vector de Fisher y usar search_similar_images para encontrar las imágenes más similares.
    """
    image_sift_descriptors = load_descriptors("assets//data//SIFT.pkl")
    image_sift_keypoints = load_keypoints("assets//data//SIFT.pkl")
    # images_sift = load_descriptors("result//data//sunflower//SIFT.pkl")
    # gmm = load_gmm_model("result//model//gmm_model_fisher_vector.pkl")
    # image_descriptors = image_sift[0][2]
    # sift_descriptors = [item[2] for item in images_sift]
    # print(type(sift_descriptors))
    # print(len(sift_descriptors))
    # print(sift_descriptors[6][2])
    # images_descriptors = np.array(sift_descriptors)
    # print(images_descriptors.shape)
    # M = 4
    # Ks = [256] * M  # Number of centroids for each sub-vector
    # # Split data into sub-vectors
    # sub_vectors = split_data(images_descriptors, M, Ks)
    # # Train codebooks
    # codebooks = train_codebooks(sub_vectors, Ks)
    # # Quantize data
    # codes = quantize_data(images_descriptors, codebooks)


if __name__ == "__main__":
    main()
