import pickle

import numpy as np

from algorithms_keypoints_descriptors import execSift, readImage
from fisher_matrix import loadImages, getFisherVector
from my_faiss import buildIndex, loadIndex, saveIndex
from my_neo4j import Neo4jClient, Neo4jServer

index_file = "result/faiss_index.index"
model_file = "result/the_choosen_one_k_nodes_64_hyperparameter_3.pkl"
neo4j_server = Neo4jServer(
    hostname="localhost",
    port=7687,
    username="neo4j",
    password="your_password",
    database="neo4j",
)


def preProcess():
    images, _ = loadImages("asset/flower/train")
    with open(model_file, "rb") as f:
        best_svm = pickle.load(f)

    nodes = list()
    fisher_vectors = list()
    for img in images:
        if img.descriptors is not None:
            des =  np.array(img.descriptors)
            fisher_vector = getFisherVector(des, best_svm)
            fisher_vectors.append(fisher_vector)
            nodes.append({"filename": img.filename, "vector": fisher_vector})

    with Neo4jClient(neo4j_server) as client:
        client.insertMany(nodes, node_label="MyNode")
    vectors = np.array(fisher_vectors)
    faiss_index = buildIndex(vectors)
    saveIndex(faiss_index, index_file)


def main():
    query_image = readImage("asset/flower/test/na/Image_1.jpg", 80, 80)
    _, query_img_des = execSift(query_image)

    with open(model_file, "rb") as f:
        best_svm = pickle.load(f)
    query_fisher = getFisherVector(query_img_des, best_svm)

    faiss_index = loadIndex(index_file)

    distances, indices = faiss_index.search(query_fisher.reshape(1, -1), 5)
    similar_images = []
    for idx in indices[0]:
        if idx != -1:
            with Neo4jClient(neo4j_server) as client:
                result = client.session.run(
                    "MATCH (img:MyNode) WHERE id(img) = $id RETURN img.filename",
                    id=idx,
                )
                for record in result:
                    similar_images.append(record)
    print(similar_images)


if __name__ == "__main__":
    # preProcess()
    main()
