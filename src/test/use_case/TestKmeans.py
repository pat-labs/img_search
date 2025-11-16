import os
import unittest

import numpy as np

from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.Kmeans import KMeans
from src.main.domain.Knodes import KNodes
from src.main.presentation.ImageDescriptorAnalyzer import DescriptorType
from src.test.resourse.DatasetMock import DatasetMock


class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = DatasetMock.animals_mock()
        self.model_dir = DatasetMock.get_mock_dir()

    def test_fit_predict_and_save(self):
        k_nodes = KNodes.K16
        descriptor_type = DescriptorType.AKAZE

        images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)

        kmeans = KMeans(k=k_nodes, descriptor_type=descriptor_type)
        kmeans.fit(image_descriptor_data)

        self.assertIsNotNone(kmeans.cluster_centers_)
        # AKAZE descriptors are 61-dim
        self.assertEqual(kmeans.cluster_centers_.shape, (k_nodes.value, 61))
        self.assertIsNotNone(kmeans.labels_)
        self.assertGreater(kmeans.inertia_, 0)

        query_image = images_data[0]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        predictions = kmeans.predict(desc)
        self.assertEqual(len(predictions), len(desc))
        self.assertTrue(all(0 <= p < k_nodes.value for p in predictions))

        model_path = kmeans.save_model(self.model_dir)
        self.assertTrue(os.path.exists(model_path))

        kmeans_loaded = KMeans.load_model(model_path)
        self.assertIsNotNone(kmeans_loaded)
        self.assertEqual(kmeans_loaded.k, k_nodes)
        self.assertIsNotNone(kmeans_loaded.cluster_centers_)
        np.testing.assert_array_equal(kmeans.cluster_centers_, kmeans_loaded.cluster_centers_)


if __name__ == '__main__':
    unittest.main()
