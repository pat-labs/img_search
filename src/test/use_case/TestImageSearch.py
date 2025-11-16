import unittest
import numpy as np

from src.main.application.use_case.FaissIndex import FaissIndex
from src.main.application.use_case.FisherMatrix import FisherMatrix, KNodes
from src.main.application.use_case.ImageUtil import ImageUtil, ImageDataFeature
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType
from src.test.resourse.DatasetMock import DatasetMock


class TestImageSearch(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = DatasetMock.animals_mock()

    def test_image_search_workflow(self):
        descriptor_type = DescriptorType.ANSIOTROPIC_SIFT
        classifier_type = ClassifierType.SVM
        images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)

        # 1. Train the GMM part of the Fisher Matrix model
        fm = FisherMatrix(
            k=KNodes.K16,
            descriptor_type=descriptor_type,
            classifier_type=classifier_type
        )
        # We only need to train the GMM on a global set of descriptors
        all_descriptors = np.vstack([item.descriptor for item in image_descriptor_data if item.descriptor is not None])
        fm._train_gmm(all_descriptors)
        
        # 2. Build dataset vectors using the Spatial Pyramid Matching recipe
        print("Building Spatial Pyramid Fisher Vectors for the dataset...")
        dataset_vectors = [FisherMatrix.compute_spatial_fisher_vector(item, fm) for item in image_descriptor_data]
        dataset_vectors_np = np.array(dataset_vectors, dtype=np.float32)
        metadata = [item.path for item in image_descriptor_data]

        # 3. Build the Faiss index
        faiss_index = FaissIndex()
        faiss_index.build_index(dataset_vectors_np, metadata)
        self.assertIsNotNone(faiss_index.index)

        # 4. Create a query vector for a test image using the same SPM recipe
        query_image_data = image_descriptor_data[0]
        query_vector = FisherMatrix.compute_spatial_fisher_vector(query_image_data, fm)

        # 5. Search the index
        results = faiss_index.search(query_vector)

        # 6. Print and assert the results
        hyperparameters = f"[Query: {query_image_data.path}]\n"
        header = "Matched, Distance, Found\n"
        report = hyperparameters + header
        for idx, res in enumerate(results):
            report += f"{idx}, {res[0]}, {res[1]:.4f}, {1 if res[0] == query_image_data.path else 0}\n"
        print(report)
        
        # The top result should be the image itself
        self.assertEqual(results[0][0], query_image_data.path)


if __name__ == '__main__':
    unittest.main()
