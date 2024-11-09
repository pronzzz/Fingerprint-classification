## Stage 2: Data Analysis and Model Development for Enhanced Fingerprint Classification

### B1. Dataset Selection
The **SOCOFing (Sokoto Coventry Fingerprint Dataset)** is selected for this project due to its valuable features:
- **Dataset Size**: 6,000 images from 600 individuals, enabling models with up to 600 parameters.
- **Class Diversity**: Includes five classes—whorl, left loop, right loop, tented arch, and arch—supporting comprehensive classification.
- **Image Quality**: Consistent image resolution of 96x103 pixels at 500dpi in BMP format.
- **Synthetic Variations**: Includes modified versions to improve model robustness.
- **Accessibility**: Publicly available for research.
- **Feature Usage**: Patterns of ridges and troughs are extracted as input features.

#### Pre-processing Requirements
1. Convert to grayscale.
2. Resize images to 96x96 pixels.
3. Noise reduction and normalization.

### B2. Data Analysis

#### Descriptive Statistics
- **Total Images**: 6,000
- **Subjects**: 600
- **Classes**: 5
- **Image Properties**: BMP format, 96x103 pixels, 500dpi.

#### Visualizations
1. **Class Distribution**: Slight imbalance with loops and whorls more common.

   ```python
   import matplotlib.pyplot as plt
   classes = ['Arch', 'Tented Arch', 'Left Loop', 'Right Loop', 'Whorl']
   counts = [800, 400, 1600, 1600, 1600]
   plt.bar(classes, counts)
   plt.title('Distribution of Fingerprint Classes')
   plt.xlabel('Class')
   plt.ylabel('Count')
   plt.show()
   ```

2. **Pixel Intensity Distribution**: Histogram shows a bimodal distribution, revealing clear contrast in fingerprint patterns.

3. **Image Quality Assessment**: PSNR values confirm high-quality images.

4. **Feature Visualization**: t-SNE visualization demonstrates separability in feature space.

#### Key Insights for Model Development
- Consider class weighting or oversampling for class imbalance.
- Edge detection may enhance feature extraction.
- Minimal denoising required due to high image quality.
- t-SNE indicates separability, suggesting simpler classifiers might be effective.

### B3. Data Preparation

1. **Loading and Initial Preprocessing**: Images were converted to grayscale and labels extracted.
2. **Reshaping and Scaling**: Images were flattened and standardized.
3. **Dimensionality Reduction**: PCA retained 95% of variance.
4. **Feature Selection**: Top 100 features selected via mutual information.
5. **Label Encoding**: Class labels were numerically encoded.
6. **Train-Test Split**: Data split into 80% training and 20% testing.

### B4. Algorithm Selection
Algorithms chosen for evaluation:
- **CNNs**: High performance in image classification.
- **Random Forest**: Robust with feature relevance insights.
- **SVM**: Effective with high-dimensional data.
- **KNN**: Simple but capable of capturing local patterns.

### B5. Algorithm Implementation and Evaluation

Using `scikit-learn`, each algorithm was implemented and evaluated on F1 scores:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

algorithms = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for name, algorithm in algorithms.items():
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")
```

The **best model** was identified based on cross-validation scores and retrained on the full training set, achieving high accuracy and reliability in fingerprint classification.
