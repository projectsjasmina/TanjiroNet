# 🥒 TanjiroNet: Efficient Plant Pathology (Cucumber Disease Classification)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Params](https://img.shields.io/badge/Parameters-0.58M-blue)
![F1-Score](https://img.shields.io/badge/Macro_F1-0.82-green)

## 📄 Project Overview
This project implements **TanjiroNet**, a custom lightweight Convolutional Neural Network (CNN) designed for efficient plant disease identification in cucumbers. The goal was to create a mobile-ready model that achieves high accuracy with minimal computational cost.

The model classifies cucumber images into **8 categories** (disease types and one 'Fresh Cucumber' class) and achieved a final **Macro F1-Score of 0.8172**.

## 🧠 Model Architecture: TanjiroNet
TanjiroNet is a custom architecture highly inspired by **MobileNetV3** principles, focusing on parameter efficiency and speed.

**Key Design Choices:**
*   **Ultra-lightweight:** The model contains only **0.58 Million parameters**.
*   **Depthwise Separable Convolutions:** Used in both the `DepthwiseSeparableStem` and `Bottleneck` blocks to drastically reduce the number of floating-point operations (FLOPS) and optimize memory usage.
*   **Inverted Residual Bottlenecks:** Implemented for efficient processing and superior gradient flow, a core component for tiny, high-performing networks.
*   **Hardswish Activation:** Used throughout the network for a balance between speed and performance.

## 🛠️ Advanced Training & Evaluation Techniques
To ensure the small model generalized well and resisted real-world image noise, the following advanced techniques were employed:

1.  **Cosine Annealing Learning Rate Scheduler:** Utilized over a two-stage training process (30 + 30 epochs) to smoothly reduce the learning rate, helping the model converge to a better minimum.
2.  **Test-Time Augmentation (TTA):** Predictions were averaged across multiple augmented views (flips, rotations) of the validation images to improve model **robustness** and increase the final F1-score.
3.  **Weight Averaging (SWA-like):** Averaged the state dictionaries of the top-performing checkpoints to further enhance generalization.
4.  **Label Smoothing:** Used with `nn.CrossEntropyLoss` to regularize the model and prevent overconfident predictions.

## 📊 Performance Summary (Standard Validation)
| Metric | Value |
| :--- | :--- |
| **Macro F1-Score** | **0.8172** |
| Accuracy | 0.8256 |
| Total Loss | 1.4570 |
| Total Images | 1289 |
| Number of Classes | 8 |

## ⚙️ Tech Stack & Dependencies
*   **Deep Learning:** PyTorch, Torchvision
*   **Data Handling:** NumPy, Pandas, PIL
*   **Evaluation:** Scikit-learn (for `f1_score`, `confusion_matrix`, `classification_report`)

### 🚀 How to Run
1.  Clone the repository.
2.  Install required libraries (if not already present):
    ```bash
    pip install torch torchvision matplotlib seaborn scikit-learn pandas
    ```
3.  Execute the Jupyter Notebook `Cucumber Disease Classification Model.ipynb`.
