# Main

Optimizing Efficiency in Large-Scale Image Retrieval Using Inverted File with Approximate Distance Computation (IVFADC)

This paper delves into optimizing efficiency for large-scale image retrieval by leveraging the Inverted File with Approximate Distance Computation (IVFADC) method for compact image representations. The study explores the co-optimization of representation, dimensionality reduction, and indexing algorithms to enhance search accuracy while effectively managing memory and computational resource utilization. By incorporating local descriptors using a descriptor derived from Bag-of-Words (BOF) techniques and the Fisher kernel, the research aims to achieve scalable search precision. The IVFADC approach proves to be a promising solution for achieving efficient and accurate large-scale image retrieval.

Key Points:

The study investigates the optimization of image representation, dimensionality reduction, and indexing algorithms for large-scale image retrieval.
The IVFADC method is employed to create compact image representations for efficient search.
Local descriptors derived from BOF techniques and the Fisher kernel are incorporated to enhance search accuracy.
The IVFADC approach demonstrates promising results for achieving scalable and efficient image retrieval.
Additional Notes:

The paper highlights the importance of balancing search accuracy with resource efficiency in large-scale image retrieval systems.
The IVFADC method offers a promising approach for achieving this balance, particularly for large-scale image collections.
The research contributes to the ongoing efforts to develop efficient and accurate image retrieval algorithms.

## TO DO

### Dataset

- [] (landscape)[https://www.kaggle.com/datasets/arnaud58/landscape-pictures]
- [] (flowers)[https://www.kaggle.com/datasets/imsparsh/flowers-dataset]
- [] (Human Faces)[https://www.kaggle.com/datasets/ashwingupta3012/human-faces]

## State of art

### Base

- [] SIFT Scale-Invariant Feature Transform
    -[] (SIFT Detector)[https://www.youtube.com/watch?v=ram-jbLJjFg]
- [] ORB Oriented FAST and Rotated BRIEF
- [] SIFT vs ORB
    - [] Benchmarking

### Search

- [] IVF inverted file index
       - [] IVFADC
           (Facebook AI and the Index Factory)[https://www.pinecone.io/learn/series/faiss/composite-indexes/]
       - [] IVFPQ product quantization (PQ)
           (Similarity Search with IVFPQ)[https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3]
           (Product Quantizer Aware Inverted Index for Scalable Nearest Neighbor Search[https://openaccess.thecvf.com/content/ICCV2021/papers/Noh_Product_Quantizer_Aware_Inverted_Index_for_Scalable_Nearest_Neighbor_Search_ICCV_2021_paper.pdf]

## Docs

- (SIFT Algorithm: How to Use SIFT for Image Matching in Python)[https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/]
- (Ficha Inscripcion Tesis)[https://docs.google.com/document/d/1dg4DF4Its7Sb1FxQr0ni6jsnYDoYubK_/edit]
- (SIFT Detector)[https://www.youtube.com/watch?v=IBcsS8_gPzE]


## knowlegde

- [] region segmentation
- [] sift map reduce
- [] lsh  buckets