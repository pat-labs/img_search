# An Anisotropic SIFT–Based Multi-Stage Coarse-to-Fine Image Retrieval Framework with Cross-Domain Evaluation

We propose a coarse-to-fine image retrieval framework that combines anisotropic SIFT descriptors with pyramidal Spatial Fisher Vector (SFV) representations for scalable and accurate visual search on large image collections. Local features are extracted using anisotropic SIFT in a diffusion-based scale space that preserves edges while suppressing noise, producing stable and discriminative descriptors. A global descriptor pool is used to train Gaussian Mixture Models (GMMs) with multiple vocabulary sizes, from which multi-level SFVs are computed over increasingly fine spatial grids.
To balance accuracy and efficiency, the framework employs a multi-stage retrieval strategy. Coarse SFVs with small vocabularies and coarse grids first retrieve an initial candidate set. Medium and fine stages then use larger vocabularies and finer spatial pyramids to re-rank only these candidates, significantly reducing expensive high-dimensional comparisons. Finally, RANSAC-based geometric verification is applied to top-ranked images to enforce geometric consistency and suppress false positives.
Experiments on Oxford Flowers, cancer-cell imagery, and apparel datasets demonstrate that the pyramidal coarse-to-fine design reduces end-to-end computation by restricting fine-grained comparison and geometric verification to a small candidate subset, while maintaining and improving retrieval precision. These results confirm the robustness and scalability of the proposed approach for modern computer vision systems and large-scale image retrieval applications.The rapid growth of digital imagery—driven by high-resolution cameras, large-scale sensing infrastructures, and the proliferation of social media platforms—has created an unprecedented demand for efficient and accurate visual search systems. Modern applications in domains such as e-commerce, biomedical imaging, and environmental monitoring increasingly depend on automated image analysis to interpret, classify, and retrieve visual content from massive repositories. As the volume, diversity, and resolution of visual data continue to increase, conventional feature-based or manually curated retrieval schemes no longer scale in terms of computation, storage, or retrieval quality. This motivates the design of more scalable, discriminative, and computationally efficient image representations and retrieval pipelines.

State-of-the-art image retrieval and recognition pipelines typically rely on local descriptors such as SIFT and its variants, deep feature embeddings, or hybrid representations that combine statistical modeling with spatial encoding. However, many application domains require descriptors that are simultaneously robust to noise, sensitive to fine-grained structural variations, and computationally tractable at scale. These requirements are particularly pronounced in several critical use cases. In botanical research, automated flower identification supports large-scale species cataloging and ecological monitoring directly from in-the-wild photographs. In medical imaging, especially cancer diagnostics, precise visual pattern matching enables the detection of malignant regions by comparing pathology slides against curated reference sets, assisting early diagnosis and reducing expert workload. In e-commerce, fashion-oriented social media has shifted consumer behavior toward image-based product discovery, where users expect to retrieve apparel visually similar to items seen on influencers or public figures using only a query image.

Despite significant progress in deep learning and feature design, maintaining high retrieval precision while scaling to millions of images remains challenging, particularly for subtle textures, complex biological morphology, or rapidly evolving product catalogs. These challenges motivate multi-stage retrieval frameworks that trade off speed and accuracy in a principled way. The proposed framework addresses this need by combining anisotropic local feature extraction, pyramidal Spatial Fisher Vector modeling, and hierarchical coarse-to-fine retrieval with geometric verification to deliver scalable, fine-grained visual search across heterogeneous domains.Content-based image retrieval (CBIR) traditionally relies on hand-crafted local descriptors, with SIFT being the most influential due to its robustness to scale, rotation, and illumination variation [1]. Classical retrieval pipelines built on SIFT and its variants perform keypoint detection, descriptor extraction, and nearest-neighbor matching, forming the basis of early systems such as Video Google [2]. SURF further improved computational efficiency with Hessian-based detection and integral image approximations [3]. However, the isotropic Gaussian scale-space used in standard SIFT can blur fine structures, motivating adaptive and anisotropic variants designed to preserve edges.

To address the limitations of raw keypoint matching, feature aggregation methods such as Fisher Vectors (FVs) [4] and Spatial Pyramid Matching (SPM) [5] encode local descriptors into compact but highly discriminative global signatures. FVs model descriptor distributions using Gaussian Mixture Models, while spatial pyramids introduce multi-level grid structures that capture coarse and fine geometric relationships. These techniques remain foundational for large-scale retrieval systems due to their strong discriminative power.

Scalable indexing of high-dimensional vectors has advanced significantly through approximate nearest neighbor (ANN) search. Faiss [6], along with product quantization and hierarchical indexing schemes, enables billion-scale similarity search with low latency. Coarse-to-fine strategies using multiple indexes or multi-level vocabularies have demonstrated strong efficiency by reducing candidate sets before applying expensive verification [2], [7].

Geometric verification plays a critical role in high-precision retrieval. RANSAC [8] and improved variants such as LO-RANSAC [9] enforce geometric consistency by estimating transformations between matched keypoints, significantly reducing false positives in object retrieval pipelines.

Deep learning has further transformed image retrieval by enabling models to learn global embeddings directly from data. CNN-based aggregation [10], deep convolutional feature weighting [11], and large-scale retrieval surveys [12] highlight major progress in representation learning. In fashion retrieval, datasets such as DeepFashion [13] and models for visual compatibility, cross-domain matching, and attribute learning [14], [15] have significantly improved semantic-level understanding. Although effective for category-level retrieval, many deep models struggle with fine-grained geometric distinctions required for exact instance matching.

Despite these advances, a gap remains in systems that unify robust local descriptors, discriminative spatial aggregation, scalable ANN search, and precise geometric verification. Hierarchical coarse-to-fine pipelines, combined with powerful feature representations, provide a promising direction for balancing accuracy and scalability. Our system builds directly on this body of work by integrating anisotropic SIFT-based features, spatial Fisher Vectors, multi-index Faiss retrieval, and RANSAC verification into a unified architecture optimized for high-precision visual matching.The study adopts a structured research design centered on the development and evaluation of a multi-stage, coarse-to-fine image retrieval system optimized for apparel search. The design integrates offline model construction—where hierarchical visual vocabularies and descriptors are generated—and an online retrieval pipeline that progressively filters candidates with increasing precision. This architecture enables systematic testing of efficiency and accuracy across large-scale image collections.

Data samples were drawn from a publicly available apparel dataset containing annotated product images. Rather than using a human participant sample, the study relies on automated sampling of visual inputs, ensuring that all images undergo identical preprocessing steps. The dataset is processed to extract \textbf{anisotropic SIFT features}, a variant chosen for its robustness to the textures, edges, and fine structural variations characteristic of clothing. All images are included in the extraction pipeline, removing selection bias and ensuring comprehensive coverage of apparel categories.

Information is gathered through a two-phase data collection procedure. First, in the offline phase, each image is processed using anisotropic diffusion to compute local descriptors, which are then aggregated to train \textbf{Gaussian Mixture Models (GMMs)} of multiple vocabulary sizes. \textbf{Spatial Fisher Vectors (SFVs)} are generated using multi-level spatial pyramids—ranging from coarse $2\times2$ grids to fine $8\times8$ grids—to encode both local appearance and spatial layout. These descriptors are indexed using \textbf{Faiss} to create multiple searchable representations with varying levels of complexity. Second, in the online phase, a query image undergoes the same extraction and encoding process. Its descriptors are submitted to a sequence of Faiss indexes in a coarse-to-fine manner: a coarse index retrieves broad candidates, medium-level descriptors refine the ranking, and fine-level descriptors capture detailed structural variations.

The collected descriptor data is analyzed through a hierarchical pipeline designed to balance speed and accuracy. Distances between SFVs are computed using Faiss’s product-quantized nearest neighbor search. Candidate sets are iteratively reduced—from approximately 100 coarse matches to 20 medium matches and finally to a top subset re-ranked by fine-grained descriptors. To ensure geometric validity, \textbf{RANSAC-based homography estimation} is applied to the highest-ranked results, filtering feature correspondences and computing inlier counts as the final similarity score. This analysis procedure quantifies both global similarity and local geometric consistency.

Ethical considerations are addressed by relying exclusively on public, non-sensitive image datasets containing product photos rather than personal or identifying images of individuals. No personal data is processed, no user profiles are involved, and no private behavioral information is collected or inferred. All experiments adhere to responsible use of publicly shared content and respect licensing constraints associated with open datasets.

The methodological choices are justified by the requirements of large-scale apparel retrieval. Anisotropic SIFT is selected for its superior handling of fine fabric structures, while Fisher Vectors and spatial pyramids provide a compact yet highly discriminative representation. Multiple vocabulary sizes and indexing levels enable efficient catalog-wide search without sacrificing the ability to capture subtle geometric nuances. Finally, limiting RANSAC verification to top-ranked candidates ensures that geometric precision is achieved with minimal computational overhead. This combination offers a balanced and empirically grounded approach to high-performance visual search in the context of modern, image-driven e-commerce behavior.## Dataset

**Table I**  
**Clothing Dataset (Full): Summary of Dataset Parameters**

| Parameter                 | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| Dataset Name              | Clothing Dataset (Full)                                             |
| Total Samples             | ~5,000 images                                                       |
| Number of Classes         | 20 apparel categories                                               |
| Image Resolution          | High-resolution product images (varies by item, e-commerce style)   |
| Annotation File           | `images.csv` with image ID, class label                             |
| Label Types               | Multiclass (20 classes)                                             |
| File Format               | JPEG images                                        |
| Visual Variability        | Variation in texture, pattern, shape, lighting, and viewpoint       |
| Domain                    | E-commerce product photography                                      |
| License                   | CC0-1.0 (Public Domain)                                             |
| Suitable Tasks            | Image retrieval, classification, feature analysis, clustering       |
| Reason for Selection      | Balanced size, realistic imagery, free license, strong visual detail |

**Table II**  
**Class Distribution in the Clothing Dataset (Full)**

| Class        | Samples |
|--------------|---------|
| T-Shirt      | 1011    |
| Long Sleeve  | 699     |
| Pants        | 692     |
| Shoes        | 431     |
| Shirt        | 378     |
| Dress        | 357     |
| Outwear      | 312     |
| Shorts       | 308     |
| Hat          | 171     |
| Skirt        | 155     |
| Polo         | 120     |
| Undershirt   | 118     |
| Blazer       | 109     |
| Hoodie       | 100     |
| Body         | 69      |
| Top          | 43      |
| Blouse       | 23      |

The dataset offers a realistic representation of modern e-commerce product imagery and provides sufficient category variety and visual richness to evaluate hierarchical image descriptors and coarse-to-fine retrieval strategies. Its balanced scale allows meaningful experiments without excessive computational overhead, and its public-domain license ensures full reproducibility.


**Table 3**  
**Multi Cancer Dataset (Kaggle) — Summary of Characteristics**

| Parameter                 | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| Dataset Name              | Multi Cancer Dataset (Obuli Sai Naren)                              |
| Total Images              | ~130,000 histopathology images                                      |
| Number of Cancer Types    | 4 major cancer types (Cervical, ALL Leukemia, Brain, Lung/Colon)   |
| Sub-Classes               | Multiple histopathological subtypes within each cancer category     |
| Image Format              | JPEG                                                                 |
| Image Resolution          | 512 × 512 pixels                                                     |
| Label Types               | Multiclass labels for cancer type and subtype                       |
| Domain                    | Medical histopathology (microscope tissue slides)                   |
| Annotation Source         | Directory-based class labeling                                      |
| Typical Tasks             | Classification, feature extraction, segmentation, generalization     |
| License / Availability    | Publicly available via Kaggle                                       |
| Visual Characteristics    | High intra-class variation; stained tissue slides                   |

**Table 4**  
**Class Labels and Experimental Motivation**

| Class / Label Group        | Description                                                        | Reason for Choosing in Experiments                                 |
|----------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------|
| Cervical Cancer            | Tissue slides of cervical epithelial abnormalities                 | Strong texture variation useful for robustness testing              |
| ALL (Acute Lymphoblastic Leukemia) | Blood smear histology images with lymphoblast patterns      | Good benchmark for fine-grained feature differentiation             |
| Brain Tumor Tissues        | Includes glioma, meningioma, and pituitary tumor subtypes         | Tests descriptor ability to discriminate subtle structural details  |
| Lung & Colon Cancer        | Histopathology slides with glandular and cellular patterns         | High intra-class diversity challenges coarse-to-fine ranking        |
| Multiple Subtypes per Class| Further division of each cancer type                               | Supports evaluation of hierarchical classifiers or retrieval stages |
| Uniform 512×512 Format     | Fixed-size input for all images                                    | Ensures fair comparison across descriptors and retrieval models     |
| Large Sample Size (~130k)  | Sufficient images per class                                        | Enables statistically reliable performance measurement              |
| Public Availability        | Free on Kaggle                                                     | Ensures reproducibility for IEEE research                           |

The Multi Cancer Dataset provides a comprehensive and realistic collection of histopathological imagery spanning multiple cancer types and subtypes. Its large scale and high-resolution microscope slides capture rich cellular, structural, and textural variations essential for evaluating feature extraction and discriminative learning methods. The diversity of tissue patterns across cancer categories supports the assessment of fine-grained and coarse-level classification performance, making the dataset appropriate for testing hierarchical representations and multi-stage decision pipelines. Furthermore, the uniform image format and public availability through Kaggle ensure consistent preprocessing, reproducibility, and accessibility for the research community.

**Table 5**  
**Flowers Dataset (Kaggle) — Summary of Characteristics**

| Parameter               | Description / Value                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| Dataset Name            | Flowers Dataset (Kaggle)                                                             |
| Total Images            | ~4,000–4,500 (≈ 4,242 images) :contentReference[oaicite:1]{index=1}                      |
| Number of Classes       | 5 flower categories: Daisy, Dandelion, Rose, Sunflower, Tulip :contentReference[oaicite:2]{index=2} |
| Image Format            | JPEG                                                                                 |
| Typical Image Size      | Around 320 × 240 pixels (varies) :contentReference[oaicite:3]{index=3}                   |
| Label Type              | Multiclass labels (flower species)                                                   |
| Organization            | Images organized in subfolders per class (folder name = label) :contentReference[oaicite:4]{index=4} |
| Visual Variability      | Variation in lighting, background, viewpoint, natural variation in flowers           |
| Typical Use Cases       | Classification, feature extraction, object recognition, fine-grained visual tasks    |
| License / Availability  | Publicly available on Kaggle (free to download) :contentReference[oaicite:5]{index=5}    |

**Table 6**  
**Class Labels and Motivation for Experimental Use**

| Class (Flower Type) | Approx. Number of Images* | Why It’s Suitable for Experimental Use |
|---------------------|----------------------------|----------------------------------------|
| Daisy               | ~760                       | Provides texture and petal-edge variations, useful for testing descriptor sensitivity to fine shape detail. :contentReference[oaicite:6]{index=6} |
| Dandelion           | ~1,050                     | High intra-class variation (petal count, orientation, background), good for evaluating robustness of matching methods. :contentReference[oaicite:7]{index=7} |
| Rose                | ~780                       | Complex petal structures and overlapping shapes — tests ability to capture subtle visual structure differences. :contentReference[oaicite:8]{index=8} |
| Sunflower           | ~730                       | Distinct radial symmetry and texture; useful for evaluating spatial descriptor performance on symmetrical patterns. :contentReference[oaicite:9]{index=9} |
| Tulip               | ~980                       | Simple, clean shapes and consistent backgrounds — good for baseline evaluation and cross-class discrimination. :contentReference[oaicite:10]{index=10} |
| Balanced Class Spread | 5 classes with hundreds per class | Enables statistical significance in experiments; ensures that evaluation isn't dominated by a few classes |
| Diverse Visual Conditions | Varied backgrounds, lighting, flower orientations | Tests robustness of retrieval / descriptor methods under realistic variability |
| Manageable Size     | ~4,000 images total         | Practical for research experiments, allows fast iteration and testing without high computational cost |

The Flowers Dataset offers a compact but diverse collection of natural flower images drawn from 5 distinct species — Daisy, Dandelion, Rose, Sunflower, and Tulip — with hundreds of samples per class. Its modest total size (~4,200 images) makes it practical for rapid experimentation, while the variety in background, lighting, orientation, and flower morphology provides sufficient visual complexity to test and compare descriptor-based image retrieval or classification systems. The balanced distribution across classes and publicly accessible license facilitate reproducibility and fair evaluation. For tasks such as fine-grained visual matching, feature-extraction robustness, or retrieval sensitivity to shape and texture, the dataset represents a useful benchmark that combines manageable scale with real-world variation.


The three datasets used in this study represent distinct domains and visual characteristics, allowing the proposed retrieval framework to be evaluated across consumer, medical, and natural-image conditions. The Clothing Dataset (Full) provides high-quality e-commerce product images with rich texture, pattern, and structural variation across 20 apparel categories, making it suitable for assessing fine-grained similarity and hierarchical descriptors in real-world shopping scenarios. In contrast, the Multi Cancer Dataset offers over 100,000 histopathological images with strong intra-class variability and complex cellular morphologies, enabling rigorous testing of discriminative power and robustness in highly detailed, domain-specific visual environments. The Flowers Dataset, while smaller in scale, captures diverse natural variability in shape, color, illumination, and background across five flower species, serving as an effective benchmark for evaluating descriptor sensitivity to organic visual structures. Together, these datasets provide complementary perspectives: structured product imagery for geometric consistency, medical imagery for micro-texture discrimination, and natural imagery for variability and generalization. This combination ensures a comprehensive assessment of the system’s performance across heterogeneous visual domains.

## Process

1. [Pre process](/doc/diagram/SequencePreProcess.md)
2. [Query Image Retrievel](/doc/diagram/SequenceQueryImage.md)

## Evaluation Metrics
**Table 7**  
**Evaluation Metrics Description**

| Metric        | Description                                                                                     |
|---------------|-------------------------------------------------------------------------------------------------|
| **Variant**   | Identifies the algorithmic configuration or descriptor type used (e.g., SIFT variant, SFV level). |
| **Keypoints** | Number of local interest points detected in the image; indicates available structural information and affects computational load. |
| **Match Ratio** | Proportion of successful feature matches relative to total keypoints; reflects descriptor discriminativeness and matching reliability. |
| **Avg Distance** | Average Euclidean distance between matched descriptors or Fisher Vectors; lower values indicate higher similarity between image representations. |
| **Memory (MB)** | Total memory required to store descriptors, Fisher Vectors, or model parameters; used to evaluate scalability. |
| **Time (sec)** | Execution time for feature extraction, Fisher Vector computation, ranking, or geometric verification; measures computational efficiency. |

## Results
**Table 8**  
**Summary of Experimental Results Across All Datasets**

| Dataset                | Method            | Keypoints (avg) | Match Ratio (range) | Avg Distance | Memory Usage | Time per Image | Observed Behavior |
|------------------------|-------------------|------------------|----------------------|--------------|---------------|------------------|--------------------|
| **Clothing**           | Standard SIFT     | ~1,117           | 0.06 – 0.13          | High         | Low           | ~0.004 hrs       | Fast but fragile under rotation/scale; misses fine patterns. |
|                        | Anisotropic SIFT  | ~7,792           | 0.43 – 0.75          | Low          | Very High     | ~0.20 hrs        | Highly robust; captures texture and patterns; slower. |
| **Multi Cancer**       | Standard SIFT     | ~4,674           | 0.23 – 0.28          | High         | Low           | 0.5–0.9 hrs      | Struggles with medical micro-textures; moderate performance. |
|                        | Anisotropic SIFT  | ~32,127          | 0.78 – 0.84          | Low          | Extremely High | 20–40 hrs         | Outstanding accuracy; excellent for histopathological detail; computationally prohibitive. |
| **Flowers**            | Standard SIFT     | ~1,837           | 0.10 – 0.15          | High         | Low           | ~0.041 hrs       | Sensitive to orientation/background variation. |
|                        | Anisotropic SIFT  | ~13,226          | 0.70 – 0.80          | Low          | Very High     | ~2.5 hrs         | Stable under brightness/scale; suitable for fine-grained patterns. |

The comparative evaluation between the proposed Anisotropic SIFT implementation and the baseline standard SIFT (Adhoc) across the three datasets—Clothing Dataset (Full), Multi Cancer Dataset, and Flowers Dataset—reveals a consistent and pronounced trade-off between matching robustness and computational efficiency. Although the magnitude of differences varies by domain, the overall behavioral pattern of both methods remains stable.

Across all datasets, Anisotropic SIFT identifies substantially more keypoints than the standard SIFT implementation. In the Clothing Dataset, the Anisotropic variant detects approximately seven times more keypoints (7,792 vs. 1,117), resulting in significantly improved match ratios under common transformations such as rotation, scaling, and illumination changes. This advantage extends to the Flowers dataset, where organic structures, curved edges, and texture-rich regions benefit from the dense, diffusion-based scale-space, yielding match ratios consistently above 0.70 for transformations that reduce the performance of standard SIFT below 0.15.

The Multi Cancer Dataset amplifies this trend due to its highly complex micro-textural patterns. Here, Anisotropic SIFT reaches extreme densities—32,127 detected keypoints versus 4,674 for the baseline SIFT. The high-frequency information present in histopathological slides directly benefits from anisotropic diffusion, producing match ratios as high as 0.84 under brightness variation and 0.78 under scaling. In comparison, the Adhoc SIFT only reaches 0.23–0.28 under the same conditions. These results suggest that Anisotropic SIFT is particularly suited for medical imaging scenarios, where subtle texture consistency and rotational invariance are essential for reliable matching.

However, this robustness comes with a substantial computational penalty. Across all datasets, the Anisotropic implementation consistently requires between 50× and 60× more time than the baseline SIFT. For instance, processing a single scaled clothing image takes approximately 0.2 hours with Anisotropic SIFT compared to only 0.004 hours with the Adhoc implementation. In the Multi Cancer dataset, the disparity becomes even more pronounced: the scale_up_1.5x operation takes 41.79 hours using Anisotropic SIFT, whereas the Adhoc SIFT completes the same task in under an hour (0.86 hours). Similarly, brightness and rotation transformations in the Flowers dataset show time differences of over two orders of magnitude.

Memory consumption follows the same pattern. The dense feature extraction and extended scale-space exploration of the Anisotropic pipeline result in memory usage that is typically 50× to 100× greater, especially evident in datasets with high texture density like the Multi Cancer set. In contrast, the Adhoc SIFT remains extremely lightweight, making it ideal for environments where memory is constrained.

Finally, the behavior observed in individual sample analyses reinforces these aggregate trends. For a representative image in the Clothing Dataset, Anisotropic SIFT detected 13,226 keypoints compared to 1,837 from the Adhoc method. The Anisotropic implementation preserved match ratios above 0.76 across scaling and brightness changes, while the Adhoc version fell to 0.10–0.15 under rotation and flipping—indicating significant orientation sensitivity.

Overall, the results clearly delineate two operational regimes:

1. Anisotropic SIFT — High-Fidelity Mode:

- Maximizes keypoint density and match stability 
- Highly robust under affine, radiometric, and geometric transformations 
- Best suited for medical imaging, forensic analysis, and offline fine-grained retrieval 
- High computational and memory cost restricts real-time deployment

2. Standard SIFT (Adhoc) — Real-Time Mode:

- Fast and computationally inexpensive 
- Acceptable performance on simpler structures (Flowers), moderate on Clothing 
- Struggles significantly in high-texture medical imagery 
- Appropriate for applications requiring speed over maximum recall

By testing across three visually distinct datasets, the experiments demonstrate that Anisotropic SIFT consistently outperforms standard SIFT in accuracy, especially in texture-complex domains, but at the cost of severely increased computation time and memory requirements. The choice between the two methods therefore depends on the operational constraints of the application—whether priority is given to precision or efficiency.This work presented a comparative evaluation of a custom Anisotropic SIFT implementation against a standard SIFT baseline across three visually diverse datasets: Clothing Dataset (Full), Multi Cancer Dataset, and the Flowers Dataset. The results consistently demonstrate a clear trade-off between feature robustness and computational efficiency.

Across all datasets, the Anisotropic SIFT method produced dramatically higher keypoint densities—up to an order of magnitude greater—resulting in substantially improved match ratios and lower descriptor distances. This robustness was particularly evident under affine transformations, illumination changes, and complex textural structures, with match ratios frequently exceeding 0.75 where the standard SIFT variant fell below 0.15. These findings highlight the effectiveness of anisotropic diffusion in stabilizing scale-space representation and enhancing feature localization in challenging visual environments.

However, this gain in accuracy comes with significant computational cost. The Anisotropic SIFT implementation required 50× to 60× longer processing time and consumed substantially more memory, making it impractical for real-time or resource-constrained applications. In extreme cases, such as the multi-scale analysis of histopathological slides, processing times exceeded 40 hours per image, indicating that the approach is feasible primarily for offline high-precision tasks.

Collectively, the results indicate that Anisotropic SIFT is best suited for domains where maximum feature recall and transformation invariance are critical, including medical imaging, forensic analysis, and offline fine-grained retrieval. Conversely, the standard SIFT implementation provides a lightweight, efficient alternative for scenarios where computational speed and scalability outweigh the need for maximal descriptor robustness. Future work may explore hybrid approaches that combine anisotropic diffusion selectively or adaptively to reduce computational overhead while retaining high-fidelity feature extraction.[1] D. G. Lowe, “Distinctive image features from scale-invariant  keypoints,” International Journal of Computer Vision, vol. 60, no. 2,  pp. 91–110, 2004.

[2] J. Sivic and A. Zisserman, “Video Google: A text retrieval approach to object matching in videos,” in Proc. ICCV, 2003.

[3] H. Bay, T. Tuytelaars, and L. Van Gool, “SURF: Speeded up robust features,” in Proc. ECCV, 2006.

[4] F. Perronnin, J. Sánchez, and T. Mensink, “Improving the Fisher  kernel for large-scale image classification,” in Proc. ECCV, 2010.

[5] S. Lazebnik, C. Schmid, and J. Ponce, “Beyond bags of features:  Spatial pyramid matching for recognizing natural scene categories,” in  Proc. CVPR, 2006.

[6] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” IEEE Trans. Big Data, 2019.

[7] H. Jégou, M. Douze, and C. Schmid, “Product quantization for  nearest neighbor search,” IEEE Trans. PAMI, vol. 33, no. 1, pp. 117–128,  2011.

[8] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting,” Communications of the ACM, 1981.

[9] O. Chum, J. Matas, and J. Kittler, “Locally optimized RANSAC,” in Proc. DAGM, 2003.

[10] A. Babenko and V. Lempitsky, “Aggregating deep convolutional features for image retrieval,” in Proc. ICCV, 2015.

[11] Y. Kalantidis, C. Mellina, and S. Osindero, “Cross-dimensional  weighting for aggregated deep convolutional features,” in Proc. ECCV,  2016.

[12] R. Arandjelović et al., “NetVLAD: CNN architecture for weakly supervised place recognition,” in Proc. CVPR, 2016.

[13] Z. Liu et al., “DeepFashion: Powering robust clothes recognition and retrieval,” in Proc. CVPR, 2016.

[14] R. He and J. McAuley, “VBPR: Visual Bayesian personalized ranking from implicit feedback,” in Proc. AAAI, 2016.

[15] H. Han et al., “Learning fashion compatibility with bidirectional LSTMs,” in Proc. ACM Multimedia, 2017.

[1] A. Grigorev, "Clothing dataset (full, high resolution)," *Kaggle*, [Dataset]. Available: [https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full/data](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full/data). [Accessed: Dec. 3, 2025].

[2] O. S. Naren, “Multi Cancer Dataset,” Kaggle, [Dataset]. Available: https://doi.org/10.34740/KAGGLE/DSV/3415848. [Accessed: Dec. 3, 2025].

[3] S. Gupta, “Flowers Dataset,” Kaggle, [Dataset]. Available: https://www.kaggle.com/datasets/imsparsh/flowers-dataset. [Accessed: Dec. 3, 2025].
## List of Tables

**Table I:** [Clothing Dataset (Full): Summary of Dataset Parameters](#table-i)  
**Table II:** [Class Distribution in the Clothing Dataset (Full)](#table-ii)  
**Table III:** [Multi Cancer Dataset (Kaggle) — Summary of Characteristics](#table-3)  
**Table IV:** [Class Labels and Experimental Motivation](#table-4)  
**Table V:** [Flowers Dataset (Kaggle) — Summary of Characteristics](#table-5)  
**Table VI:** [Class Labels and Motivation for Experimental Use](#table-6)  
**Table VII:** [Evaluation Metrics Description](#table-7)  
**Table VIII:** [Summary of Experimental Results Across All Datasets](#table-8)  

---

## List of Images

 **Figure 1:** [Sequence Diagram for Pre-processing](/doc/diagram/SequencePreProcess.md)  
 **Figure 2:** [Sequence Diagram for Query Image Retrieval](/doc/diagram/SequenceQueryImage.md)  