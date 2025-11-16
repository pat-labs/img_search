from dataclasses import dataclass

from src.main.application.use_case.BagOfVisualWords import BagOfVisualWords
from src.main.application.use_case.FileHandler import FileHandler
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType
from src.main.domain.Knodes import KNodes


@dataclass
class BoVWPerformance:
    k: KNodes
    descriptor: DescriptorType
    classifier: ClassifierType
    time_cost: float
    memory_cost: float
    accuracy: float


class ClusterAnalyzer:
    @staticmethod
    def _generate_bovw_report(results: list[BoVWPerformance]) -> str:
        if not results:
            return "# No results to report.\n"

        results.sort(key=lambda r: r.accuracy, reverse=True)
        report = "# Bag of Visual Words Hyperparameter Analysis Report\n\n"
        report += "| K | Descriptor | Classifier | Accuracy | Time (s) | Memory (MB) |\n"
        report += "|---|---|---|---|---|---|\n"
        for r in results:
            report += f"| {r.k.name} | {r.descriptor.name} | {r.classifier.name} | {r.accuracy:.2%} | {r.time_cost:.2f} | {r.memory_cost:.2f} |\n"
        
        print("\n--- BoVW Final Report ---")
        print(report)
        return report

    @staticmethod
    def _train_and_evaluate(bovw: BagOfVisualWords, train_data, test_data, descriptor_type: DescriptorType):
        # Train the model and measure performance
        _, perf_result = PerformanceAnalyzer().measure_performance(bovw.train, train_data)

        # Evaluate accuracy on the test set
        correct_predictions = 0
        for query_image in test_data:
            feature = ImageUtil.extract_features(query_image.path, descriptor_type)
            desc = feature.descriptor
            prediction = bovw.predict(desc)
            if prediction == query_image.label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_data) if test_data else 0.0
        return perf_result, accuracy

    @staticmethod
    def _generate_prediction_file(bovw: BagOfVisualWords, test_data, descriptor_type: DescriptorType, report_dir: str):
        k_name = bovw.k.name
        desc_name = bovw.descriptor_type.name
        clf_name = bovw.classifier_type.name
        
        predictions = "image_path,actual_label,predicted_label\n"
        for query_image in test_data:
            feature = ImageUtil.extract_features(query_image.path, descriptor_type)
            desc = feature.descriptor
            prediction = bovw.predict(desc)
            predictions += f"{query_image.path},{query_image.label},{prediction}\n"
        
        FileHandler.write_file(predictions, report_dir, f"bovw_{desc_name}_{k_name}_{clf_name}", ".csv")

    @staticmethod
    def analyze_bovw(train_data, test_data, report_dir: str) -> str:
        results = []
        for descriptor_type in [DescriptorType.ORB]:
            print(f"\n--- Analyzing Descriptor: {descriptor_type.name} ---")
            image_descriptor_data = ImageUtil.extract_descriptors_parallel(train_data, descriptor_type)
            for k in KNodes:
                for classifier_type in ClassifierType:
                    print(f"  - Testing K={k.name}, Classifier={classifier_type.name}")
                    bovw = BagOfVisualWords(
                        k=k,
                        classifier_type=classifier_type,
                        descriptor_type=descriptor_type
                    )
                    
                    perf_result, accuracy = ClusterAnalyzer._train_and_evaluate(bovw, image_descriptor_data, test_data, descriptor_type)
                    
                    results.append(BoVWPerformance(
                        k=k,
                        descriptor=descriptor_type,
                        classifier=classifier_type,
                        time_cost=perf_result.execution_time_seconds,
                        memory_cost=perf_result.memory_usage_mb,
                        accuracy=accuracy
                    ))

                    ClusterAnalyzer._generate_prediction_file(bovw, test_data, descriptor_type, report_dir)
        
        return ClusterAnalyzer._generate_bovw_report(results)

if __name__ == '__main__':
    train_dir = "/home/patrick/Documents/project/img_search/asset/dataset/train"
    test_dir = "/home/patrick/Documents/project/img_search/asset/dataset/train"
    report_dir = "/home/patrick/Documents/project/img_search/asset/report/"
    train_data = ImageUtil.load_image_data_from_folder(train_dir)
    test_data = ImageUtil.load_image_data_from_folder(test_dir)
    bovw_report = ClusterAnalyzer.analyze_bovw(train_data, test_data, report_dir)
    FileHandler.write_file(bovw_report, report_dir, "bovw_hyperparameter_report", ".md")