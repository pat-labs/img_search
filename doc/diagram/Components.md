```mermaid
graph TD
    %% Define styles for a professional, grayscale look
    classDef default fill:#fff,stroke:#333,stroke-width:1px,color:#000;
    classDef group fill:#fdfdfd,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5,color:#000;
    classDef startend fill:#f0f0f0,stroke:#000,stroke-width:2px,color:#000;
    classDef io fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef loop fill:#fff,stroke:#333,stroke-width:2px,stroke-dasharray: 3 3;

    Start((Start Script)):::startend --> A["analyzer.measure_performance() starts"];

    subgraph Setup [Offline Preparation]
        A --> B["Load Image Metadata (from .csv)"]:::io;
        B --> C["Extract Anisotropic SIFT Descriptors (for all images)"];
        C --> D["Aggregate all descriptors into a global pool"];
        D --> E{Loop: K in [16, 32, 64]};
        E --> F["Train GMM for each K"];
        F --> E;
        E --> G["Store trained GMMs"];
    end

    subgraph Search [Online Coarse-to-Fine Search]
        G --> H["Select Query Image"];

        H --> I["Stage 1: Coarse Search (K=16, 2x2)"];
        I --> J[Compute Query SFV];
        J --> K{Loop over all dataset images};
        K --> L["Compute SFV for each image (on-the-fly)"];
        L --> M["Calculate distance"];
        M --> K;
        K --> N["Sort distances & get Top 100 Candidates"];

        N --> O["Stage 2: Medium Re-ranking (K=32, 4x4)"];
        O --> P[Compute Query SFV];
        P --> Q{Loop over 100 Candidates};
        Q --> R["Compute SFV for each candidate (on-the-fly)"];
        R --> S["Calculate distance"];
        S --> Q;
        Q --> T["Sort distances & get Top 20 Candidates"];

        T --> U["Stage 3: Fine Re-ranking (K=64, 8x8)"];
        U --> V[Compute Query SFV];
        V --> W{Loop over 20 Candidates};
        W --> X["Compute SFV for each candidate (on-the-fly)"];
        X --> Y["Calculate distance"];
        Y --> W;
        W --> Z["Sort distances & get Top 10 Candidates"];
    end

    subgraph Verification [Final Verification & Reporting]
        Z --> AA["Stage 4: Geometric Verification"];
        AA --> BB["Rerank Top 10 with RANSAC Inliers"];
        BB --> CC["analyzer.measure_performance() stops"];
        CC --> DD["Print Final Ranked Report"]:::io;
        DD --> EE["Print Performance Metrics (Time, Memory)"]:::io;
    end

    EE --> End((End Script)):::startend;

    %% Apply styles
    class Setup,Search,Verification group;
    class K,Q,W loop;
```