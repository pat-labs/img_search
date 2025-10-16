from src.main.application.use_case.ClusterAnalyzer import ClusterAnalyzer


def test_main():
    dataset_dir = "/home/patrick/Documents/project/img_search/asset/dataset"
    cluster_analyzer = ClusterAnalyzer(dataset_dir)
    image_paths = [item.path for item in train_data]

if __name__ == '__main__':
    test_main()