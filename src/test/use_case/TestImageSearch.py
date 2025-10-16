import unittest

import numpy as np

from src.main.application.use_case.ImageSearch import ImageSearch


class TestImageSearch(unittest.TestCase):

    def setUp(self):
        image_a = np.zeros((50, 50), dtype=np.uint8)
        image_search = ImageSearch()
        image_util = ImageUtil()
        object_isolator = ObjectIsolator()

    def main(self):
        image_sanitized = image_util.sanitize(image_a)
        objects = object_isolator.get_isolate_objects(image_sanitized)
        results = []
        for object in objects:
            image_encode = image_search(object)
            image_result = image_search.search(image_encode, 3)
            results.append(image_result)
        return results
    
if __name__ == '__main__':
    unittest.main()