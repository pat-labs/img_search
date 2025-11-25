from src.main.application.use_case.SIFTUtil import anisotropicSiftComputeKeypointsAndDescriptors


class AnisotropicSIFT:
    def __init__(self):
        pass

    def detectAndCompute(self, image, mask=None):
        return anisotropicSiftComputeKeypointsAndDescriptors(image)
    
    def descriptorSize(self):
        return 128