import os
import unittest

from src.main.application.use_case.FisherMatrix import FisherMatrix
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.Knodes import KNodes
from src.main.presentation.ImageDescriptorAnalyzer import DescriptorType
from src.test.resourse.DatasetMock import DatasetMock


class TestFisherMatrix(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = DatasetMock.animals_mock()
        self.model_dir = DatasetMock.get_mock_dir()

    def test_train_predict_and_save(self):
        k_nodes = KNodes.K16
        descriptor_type = DescriptorType.AKAZE

        images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)

        fm = FisherMatrix(
            k=k_nodes,
            descriptor_type=descriptor_type,
            classifier_type=ClassifierType.SVM
        )
        fm.train(image_descriptor_data)
        self.assertTrue(fm.is_trained())

        query_image = images_data[0]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptors
        result = fm.predict(desc)
        self.assertEqual(result, query_image.label)

        model_path = fm.save_model(self.model_dir)
        self.assertTrue(os.path.exists(model_path))

        fm_loaded = FisherMatrix.load_model(
            model_path
        )
        self.assertIsNotNone(fm_loaded)

        query_image = images_data[1]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptors
        result = fm.predict(desc)
        self.assertEqual(result, query_image.label)


if __name__ == '__main__':
    unittest.main()
