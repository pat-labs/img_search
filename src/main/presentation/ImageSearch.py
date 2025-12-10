import unittest
import numpy as np
import cv2
import os

from src.main.application.use_case.FaissIndex import FaissIndex, FaissMetric
from src.main.application.use_case.FisherMatrix import FisherMatrix, KNodes
from src.main.application.use_case.ImageUtil import ImageUtil, ImageDataFeature
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType
from src.test.resourse.DatasetMock import DatasetMock


class ImageSearch:

    def __init__(self):
        #self.dataset_dir = DatasetMock.animals_mock()
        #self.dataset_dir = "/home/patrick/Documents/project/latex/asset/dataset/flowers/sanitize"
        self.dataset_dir = "/home/patrick/Documents/project/img_search/asset/dataset/flowers/"

    def _rerank_with_ransac(self, query_features: ImageDataFeature, candidate_features: list[ImageDataFeature],
                            min_match_count=10):
        """
        Re-ranks candidate images based on geometric verification using RANSAC.
        Returns a list of (path, inlier_count) tuples, sorted by inlier count.
        """
        reranked_results = []
        bf = cv2.BFMatcher()

        for candidate in candidate_features:
            if query_features.descriptors is None or candidate.descriptors is None:
                continue

            matches = bf.knnMatch(query_features.descriptors, candidate.descriptors, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            inlier_count = 0
            if len(good_matches) > min_match_count:
                src_pts = np.float32([query_features.keypoints[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([candidate.keypoints[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inlier_count = np.sum(mask)

            reranked_results.append((candidate.path, inlier_count))

        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results

    def test_image_search_workflow(self):
        # Base Fisher Vector size: 2 * k * descriptor_dimension = 2 * 128 * 128 = 32,768 floats
        # Spatial Grid size: 8 * 8 = 64 cells
        # Total Vector size per image: 32,768 * 64 = 2,097,152 floats. At 4 bytes per float, this is ~8.4 MB per image.
        descriptor_type = DescriptorType.ANSIOTROPIC_SIFT
        classifier_type = ClassifierType.SVM
        #images_data = ImageUtil.load_image_data_from_folder(self.dataset_dir)
        images_data = ImageUtil.load_image_data_from_csv(self.dataset_dir + "images.csv")
        base_path = self.dataset_dir + "train/"
        for item in images_data:
            item.path = base_path + item.path
        image_descriptor_data = ImageUtil.extract_descriptors_parallel(images_data, descriptor_type)

        # We only need to train the GMM on a global set of descriptors
        all_descriptors = np.vstack(
            [item.descriptors for item in image_descriptor_data if
             item.descriptors is not None and len(item.descriptors) > 0])

        # --- Setup: Create the specific models and indexes needed for the coarse-to-fine pipeline ---
        pipeline_configs = [
            # We only build an index for the coarsest level. Others are computed on-the-fly.
            {'k': KNodes.K16, 'grid': (2, 2), 'name': 'Coarse'},
            {'k': KNodes.K32, 'grid': (4, 4), 'name': 'Medium'},
            {'k': KNodes.K64, 'grid': (8, 8), 'name': 'Fine'}
        ]

        trained_models = {}
        faiss_indexes = {}

        for config in pipeline_configs:
            k = config['k']

            print(f"\n--- Training GMM for K={k.name} ---")
            fm = FisherMatrix(k=k, descriptor_type=descriptor_type, classifier_type=classifier_type)
            fm._train_gmm(all_descriptors)
            trained_models[k] = fm

        # --- Coarse-to-Fine Search Execution ---
        query_image_data = image_descriptor_data[0]

        # Stage 1: Coarse search (K=16, 2x2 grid)
        print("\n--- Stage 1: Coarse Search (K=16, 2x2 grid) - On-the-Fly ---")
        coarse_k, coarse_grid = KNodes.K16, (2, 2)
        coarse_fm = trained_models[coarse_k]
        coarse_query_vector = coarse_fm.compute_spatial_fisher_vector(query_image_data, grid_size=coarse_grid)
        coarse_results = []
        for item in image_descriptor_data:
            item_vector = coarse_fm.compute_spatial_fisher_vector(item, grid_size=coarse_grid)
            distance = np.linalg.norm(coarse_query_vector - item_vector)
            coarse_results.append((item.path, distance))
        coarse_results.sort(key=lambda x: x[1])
        coarse_candidate_paths = [res[0] for res in coarse_results[:100]]  # Take top 100
        print(f"Found {len(coarse_candidate_paths)} initial candidates.")

        # Stage 2: Medium re-ranking (K=32, 4x4 grid) - ON THE FLY
        print("\n--- Stage 2: Medium Re-ranking (K=32, 4x4 grid) - On-the-Fly ---")
        medium_k, medium_grid = KNodes.K32, (4, 4)
        medium_fm = trained_models[medium_k]
        medium_query_vector = medium_fm.compute_spatial_fisher_vector(query_image_data, grid_size=medium_grid)
        medium_rerank_results = []
        features_map = {item.path: item for item in image_descriptor_data}
        for path in coarse_candidate_paths:
            candidate_feature = features_map.get(path)
            if candidate_feature:
                candidate_vector = medium_fm.compute_spatial_fisher_vector(candidate_feature, grid_size=medium_grid)
                distance = np.linalg.norm(medium_query_vector - candidate_vector)
            else:
                distance = float('inf')
            medium_rerank_results.append((path, distance))
        medium_rerank_results.sort(key=lambda x: x[1])

        # Stage 3: Fine re-ranking (K=64, 8x8 grid) - ON THE FLY
        print("\n--- Stage 3: Fine Re-ranking (K=64, 8x8 grid) - On-the-Fly ---")
        fine_k, fine_grid = KNodes.K64, (8, 8)
        fine_candidates_count = 20
        fine_candidate_paths = [res[0] for res in medium_rerank_results[:fine_candidates_count]]
        fine_rerank_results = []
        fine_fm = trained_models[fine_k]
        fine_query_vector = fine_fm.compute_spatial_fisher_vector(query_image_data, grid_size=fine_grid)
        for path in fine_candidate_paths:
            candidate_feature = features_map.get(path)
            if candidate_feature:
                candidate_vector = fine_fm.compute_spatial_fisher_vector(candidate_feature, grid_size=fine_grid)
                distance = np.linalg.norm(fine_query_vector - candidate_vector)
            else:
                distance = float('inf')
            fine_rerank_results.append((path, distance))
        fine_rerank_results.sort(key=lambda x: x[1])

        # Stage 4 (Optional but recommended): Final geometric verification with RANSAC
        print("\n--- Stage 4: Final Geometric Verification (RANSAC) ---")
        final_candidates_count = 10  # RANSAC is expensive, so only use it on the very top results
        final_candidate_paths = [res[0] for res in fine_rerank_results[:final_candidates_count]]
        candidate_features = [features_map[path] for path in final_candidate_paths if path in features_map]
        final_results = self._rerank_with_ransac(query_image_data, candidate_features)

        # --- Final Report ---
        hyperparameters = f"[Query: {query_image_data.path}]\n"
        header = "Rank, Matched Path, RANSAC Inliers\n"
        report = hyperparameters + header
        for idx, (path, inlier_count) in enumerate(final_results[:10]):
            report += f"{idx + 1}, {os.path.basename(path)}, {inlier_count}\n"
        print(report)


if __name__ == '__main__':
    test_image_search = ImageSearch()
    analyzer = PerformanceAnalyzer()
    _, perf_result = analyzer.measure_performance(test_image_search.test_image_search_workflow)
    print(f", {perf_result.execution_time_seconds}, {perf_result.memory_usage_mb}")
