# AgroLens-AI
AgroLens-AI provides an automated diagnostic tool to identify plant diseases from leaf images. By leveraging deep learning, it enables early detection and precision treatment, reducing reliance on broad-spectrum pesticides and mitigating global yield loss.

## Getting Started

### Prerequisites
* Python 3.8 or higher

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/code-with-bhumi/AgroLens-AI.git
   cd AgroLens-AI
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
## Usage
System follows a structured 6-stage pipeline:
1. `01_image_acquisition.ipynb` to process the PlantVillage dataset.
2. `02_eda_visualization.ipynb` to analyze class distribution and image quality.
3. `03_preprocessing_pipeline.ipynb` for image normalization and augmentation.
4. `04_model_architecture.ipynb` defines the custom CNN layers.
5. `05_training_optimization.ipynb` to train the model with EarlyStopping.
6. `06_loss_analysis_overfitting_mitigation.ipynb` for performance diagnosis.

## License
Distributed under the MIT License.

**Maintained by:** [Vasi Khan](https://github.com/vasi2904k), [Bhumi Shah](https://github.com/code-with-bhumi), [Baviya](https://github.com/Baviyas)
