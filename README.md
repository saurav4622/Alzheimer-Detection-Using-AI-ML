
# Alzheimer's Disease Detection Using AI & ML

## Features 🚀
- **Dataset**: MRI scan images from [Kaggle](https://www.kaggle.com/datasets/ahmedashrafahmed/fdata-adni-dataset)
- **Tech Stack**: Python, TensorFlow/Keras, Google Colab
- **Classification Classes**:
  - AD: Alzheimer’s Disease
  - CN: Cognitively Normal
  - EMCI: Early Mild Cognitive Impairment
  - LMCI: Late Mild Cognitive Impairment
- **Optimization Algorithm**: Crow Search Algorithm (CSA)
- **Model Architecture**: Convolutional Neural Networks (CNNs)

## Dataset 📂
This project uses the [FData-ADNI Dataset](https://www.kaggle.com/datasets/ahmedashrafahmed/fdata-adni-dataset) from Kaggle. The dataset contains MRI scan images categorized into the following classes:
- **AD**: Alzheimer’s Disease
- **CN**: Cognitively Normal
- **EMCI**: Early Mild Cognitive Impairment
- **LMCI**: Late Mild Cognitive Impairment

### Download Instructions:
1. Visit the [dataset page on Kaggle](https://www.kaggle.com/datasets/ahmedashrafahmed/fdata-adni-dataset).
2. Log in to your Kaggle account.
3. Click on **Download** to get the dataset.
4. Extract the downloaded files and place them in the `dataset/` folder of this repository.

### Dataset Structure:
After placing the files, the structure should look like this:
```
dataset/
    ├── AD/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── CN/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── EMCI/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── LMCI/
        ├── image1.jpg
        └── image2.jpg
```

## Setup Instructions ⚙️
1. **Clone the repository**:
    ```bash
    git clone https://github.com/saurav4622/Alzheimer-Detection-Using-AI-ML
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the model**:
    ```bash
    streamlit run app.py
    ```

## Results 📊
| Metric    | Value |
|-----------|-------|
| Accuracy  | 95%   |
| Precision | 93%   |
| Recall    | 94%   |

## Contributors 👨‍💻
- **Sourabh** - Lead Developer

## License 📜
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
