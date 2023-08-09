class CropRatio:
    def __init__(self, ratio=(1, 1)):
        self.ratio = ratio

        # Ratio is (Width, Height)
        if ratio == (1, 1):
            self.dimensions = [(1024, 1024), (768, 768), (640, 640), (512, 512)]
        elif ratio == (3, 4):
            self.dimensions = [(960, 1280), (768, 1024), (576, 768)]
        elif ratio == (3, 2):
            self.dimensions = [(1024, 1536), (768, 1152), (640, 960), (512, 768)]
        else:
            raise Exception(
                f"Ratio not supported. Currently we only support (1, 1), (3, 4), and (3, 2).")
