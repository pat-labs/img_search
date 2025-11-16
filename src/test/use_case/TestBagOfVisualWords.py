import unittest

from src.main.application.use_case.BagOfVisualWords import BagOfVisualWords, DescriptorType, ClassifierType
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.domain.Knodes import KNodes
from src.test.resourse.DatasetMock import DatasetMock


class TestBagOfVisualWords(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = DatasetMock.animals_mock()
        self.model_dir = DatasetMock.get_mock_dir()

    def test_train_and_predict_with_save(self):
        k_nodes = KNodes.K16
        descriptor_type = DescriptorType.ORB
        classifier_type = ClassifierType.SVM

        images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)

        bovw = BagOfVisualWords(
            k=k_nodes,
            classifier_type=classifier_type,
            descriptor_type=descriptor_type
        )
        bovw.train(image_descriptor_data)
        self.assertTrue(bovw.is_trained())

        model_path = bovw.save_model(self.model_dir)

        query_image = images_data[0]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        prediction = bovw.predict(desc)
        self.assertEqual(prediction, query_image.label)

        bovw_loaded = BagOfVisualWords.load_model(model_path)
        self.assertIsNotNone(bovw_loaded)
        self.assertTrue(bovw_loaded.is_trained())

        query_image = images_data[1]
        feature = ImageUtil.extract_features(query_image.path, descriptor_type)
        desc = feature.descriptor
        prediction = bovw.predict(desc)
        self.assertEqual(prediction, query_image.label)


if __name__ == '__main__':
    unittest.main()
