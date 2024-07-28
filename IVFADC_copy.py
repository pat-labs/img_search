import copy

import faiss
import numpy as np
from skimage.feature import ORB, fisher_vector
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import linear_kernel
from sklearn.mixture import GaussianMixture

from utils import load_gmm_model, load_keypoints_and_descriptors, read_image


def reduce_dimensionality(image_descriptors, n_components=64):
    pca = PCA(n_components=n_components)
    reduced_descriptors = pca.fit_transform(image_descriptors)
    return reduced_descriptors, pca


def create_visual_vocabulary(image_descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(image_descriptors)
    return kmeans.cluster_centers_, kmeans


# Crear histogramas de palabras visuales
def create_histogram_1(image_descriptors, image_vocabulary):
    # Asignar cada descriptor al cluster más cercano
    codebook_idx = np.argmin(
        np.linalg.norm(image_vocabulary - image_descriptors, axis=1)
    )

    # Crear un histograma de tamaño igual al número de clusters
    hist = np.zeros(image_vocabulary.shape[0])
    hist[codebook_idx] = 1

    return hist


def create_histogram(image_descriptors, kmeans):
    # Predict the clusters for each descriptor
    codebook_idx = kmeans.predict(image_descriptors)

    # Create a histogram of size num_clusters
    hist, _ = np.histogram(
        codebook_idx, bins=np.arange(kmeans.n_clusters + 1), density=True
    )
    return hist


def compute_fisher_vectors(image_descriptors, gmm):
    fisher_vectors = []
    for descriptor in image_descriptors:
        fv = fisher_vector(descriptor, gmm)
        fisher_vectors.append(fv)
    return np.array(fisher_vectors)


def get_fisher_vector(descriptors):
    # TODO: For test
    # Check the number of samples
    num_samples = descriptors.shape[0]

    # Set n_components to be less than or equal to num_samples
    n_components = min(50, num_samples)

    if num_samples < 2:
        # Handle case with fewer than 2 samples by duplicating
        descriptors = np.tile(descriptors, (2, 1))

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="diag", random_state=0
    )
    gmm.fit(descriptors)
    # Inspect shapes
    print("Descriptor shape:", descriptors.shape)
    print("Means shape:", gmm.means_.shape)
    print("Covariances shape:", gmm.covariances_.shape)
    print("Weights shape:", gmm.weights_.shape)

    # Ensure shapes are compatible
    if gmm.covariances_.shape[1:] != descriptors.shape[1:]:
        raise ValueError("Shape mismatch between covariances and descriptor features.")

    fv = fisher_vector(descriptors, gmm)
    return fv


def combine_features(image_histogram, image_fisher_vector):
    combined_feature = np.concatenate((image_histogram, image_fisher_vector))
    return combined_feature


def create_faiss_index(features):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] == 0:
        raise ValueError("Features array is empty")
    print("Features shape:", features.shape)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index


# def load_and_process_query_image(query_image_path, gmm):
#     _, gray = read_image(query_image_path)
#     if gray is None:
#         raise ValueError("Failed to load or process the query image")

#     # Extract descriptors for the query image
#     detector_extractor = ORB(n_keypoints=5, harris_k=0.01)
#     detector_extractor.detect_and_extract(resize(gray, (80, 80)))
#     query_descriptors = detector_extractor.descriptors.astype('float32')

#     # Compute Fisher vector for the query image
#     query_fisher_vector = fisher_vector(query_descriptors, gmm)

#     # Ensure the query_fisher_vector has the correct shape
#     if query_fisher_vector.ndim == 1:
#         query_fisher_vector = query_fisher_vector.reshape(1, -1)

#     return query_fisher_vector


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
    _summary_
    1. Extracción de Características SIFT y Almacenamiento (formato pkl)
    2. Construcción del vocabulario visual
    * Utilizar un algoritmo de clustering como k-means para agrupar los descriptores SIFT en un número fijo de clusters. Cada cluster representa una "palabra visual".
    3. Construcción de Histogramas de Palabras Visuales
    * Para cada imagen, asignar cada descriptor SIFT al cluster más cercano (su "palabra visual").
    * Contar la frecuencia de cada palabra visual en la imagen para construir el histograma.
    4. Cálculo de Fisher Vectors
    * Calcular la matriz de Fisher basada en los histogramas.
    * Utilizar la matriz de Fisher para calcular la similitud entre pares de imágenes.
    5. Creación del Índice Faiss
    6. Búsqueda de Imágenes Similares
    _detail_
    1. Extraer características: Aplicar extract_sift_features a todas las imágenes y guardar los descriptores.
    2. Crear vocabulario visual: Concatenar todos los descriptores y usar create_visual_vocabulary para obtener los centros de los clusters.
    3. Construir histogramas: Para cada imagen, calcular el histograma utilizando create_histogram.
    4. Calcular Fisher Vectors: Utilizar compute_fisher_vectors para obtener los vectores de Fisher.
    5. Crear índice Faiss: Usar create_faiss_index para crear el índice.
    6. Buscar imágenes similares: Para una imagen de consulta, calcular su vector de Fisher y usar search_similar_images para encontrar las imágenes más similares.
    """
    data = load_keypoints_and_descriptors(
        "C://Users//pa-tr//Documents//projects//img_search//assets//data//SIFT.pkl"
    )
    gmm = load_gmm_model(
        "C://Users//pa-tr//Documents//projects//img_search//result//model//gmm_model_fisher_vector.pkl"
    )
    image_descriptors = data[0][2]

    # Perform clustering on high-dimensional descriptors
    image_vocabulary, kmeans = create_visual_vocabulary(image_descriptors, 1000)

    # Dimensionality reduction using PCA
    reduced_descriptors, pca = reduce_dimensionality(
        image_descriptors, n_components=128
    )

    # Create histograms for each image using original dimensionality descriptors
    image_histogram = create_histogram(image_descriptors, kmeans)

    # Ensure the histogram has the correct shape
    if image_histogram.ndim == 1:
        image_histogram = image_histogram.reshape(1, -1)

    image_fisher_vector = get_fisher_vector(image_histogram)

    # combined_features = combine_features(image_histogram, image_fisher_vector)
    index = create_faiss_index(image_histogram)

    ##SECOND PART
    # Load and process query image
    query_image_path = "C://Users//pa-tr//Documents//projects//img_search//assets//train//annapurna.jpg"
    _, query_gray = read_image(query_image_path)
    if query_gray is None:
        raise ValueError("Failed to load or process the query image")
    query_descriptors = copy.deepcopy(image_descriptors)
    query_reduced_descriptors = pca.transform(query_descriptors)
    query_histogram = create_histogram(query_reduced_descriptors, kmeans)

    # Ensure the query_fisher_vector is reshaped properly if needed
    if query_histogram.ndim == 1:
        query_histogram = query_histogram.reshape(1, -1)
    # Check the shape of query_fisher_vector before searching
    print("Fisher vector shape for query image:", query_histogram.shape)

    # Search for similar images
    similar_images_indices = search_similar_images(query_histogram, index)
    print("Indices of similar images:", similar_images_indices)


if __name__ == "__main__":
    main()
