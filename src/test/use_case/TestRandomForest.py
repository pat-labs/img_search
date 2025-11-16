import os
import unittest

from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.RandomForest import RandomForest
from src.main.domain.ClassifierType import ClassifierType
from src.main.presentation.ImageDescriptorAnalyzer import DescriptorType
from src.test.resourse.DatasetMock import DatasetMock


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = DatasetMock.animals_mock()
        self.model_dir = DatasetMock.get_mock_dir()

    def test_train_and_predict(self):
        descriptor_type = DescriptorType.AKAZE

        train_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(train_data, descriptor_type)

        rf_model = RandomForest(descriptor_type=descriptor_type)
        rf_model.train(image_descriptor_data)
        self.assertTrue(rf_model.is_trained())

        query_image = train_data[0]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        prediction = rf_model.predict(desc)
        self.assertEqual(prediction, query_image.label)

        saved_model_path = rf_model.save_model(self.model_dir)
        self.assertTrue(os.path.exists(saved_model_path))

        rf_loaded = RandomForest.load_model(saved_model_path)
        self.assertIsNotNone(rf_loaded)
        self.assertTrue(rf_loaded.is_trained())

        query_image = train_data[1]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        prediction = rf_loaded.predict(desc)
        self.assertEqual(prediction, query_image.label)


if __name__ == '__main__':
    unittest.main()
