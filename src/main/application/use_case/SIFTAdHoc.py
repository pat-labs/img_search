from src.main.application.use_case.SIFTUtil import siftComputeKeypointsAndDescriptors


class SIFTAdHoc:
    def __init__(self):
        pass

    def detectAndCompute(self, image, mask=None):
        return siftComputeKeypointsAndDescriptors(image)

    def descriptorSize(self):
        return 128