from src.main.application.use_case.ClusterAnalyzer import ClusterAnalyzer


def test():
    dataset_dir = "/home/patrick/Documents/project/img_search/asset/dataset"
    cluster_analyzer = ClusterAnalyzer(dataset_dir)
    cluster_analyzer.train_size()
    cluster_analyzer.test_size()

if __name__ == '__main__':
    test()