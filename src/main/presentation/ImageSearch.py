import unittest

from src.main.application.use_case.FaissIndex import FaissIndex
from src.main.application.use_case.FisherMatrix import FisherMatrix, KNodes
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType


class TestImageSearch:

    def __init__(self):
        self.dataset_dir = "/home/patrick/Documents/project/img_search/asset/dataset/sanitize"
        # Sanitized dataset
        # train_dir = "/home/patrick/Documents/project/img_search/asset/dataset/train"
        # sanitize_dir = "/home/patrick/Documents/project/img_search/asset/dataset/sanitize"
        # images_data = ImageUtil.load_image_paths_and_labels(train_dir)
        # ImageSanitizer.sanitize_dataset(images_data, sanitize_dir)

    def test_image_search_workflow(self):
        descriptor_type = DescriptorType.ANSIOTROPIC_SIFT
        classifier_type = ClassifierType.SVM
        #images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        images_data = ImageUtil.load_image_data_from_csv("/home/patrick/Documents/project/img_search/asset/dataset/train.csv")
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)
        metadata = [item.path for item in image_descriptor_data]

        fm = FisherMatrix(
            k=KNodes.K16,
            descriptor_type=descriptor_type,
            classifier_type=classifier_type
        )
        fm.train(image_descriptor_data)
        #self.assertTrue(fm.is_trained())

        faiss_index = FaissIndex()
        faiss_index.build_index(fm.fisher_vectors, metadata)
        #self.assertIsNotNone(faiss_index.index)

        query_image = images_data[0]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        query_vector = fm.compute_fisher_vector(desc)
        #k_neighbors = 100
        results = faiss_index.search(query_vector)
        hyperparameters = f"[Query: {query_image.path}]\n"
        header = "Matched, Distance, Found\n"
        report = hyperparameters + header
        for idx, res in enumerate(results):
            report += f"{idx}, {res[0]}, {res[1]:.4f}, {1 if res[0] == query_image.path else 0}\n"
        print(report)

        #self.assertIn(query_image.path, [meta for meta, _ in results])


if __name__ == '__main__':
    test_image_search = TestImageSearch()
    test_image_search.test_image_search_workflow()
