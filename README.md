## 🏷️ Palmprint Recognition: A Privacy-Focused Biometric Authentication Method

This project aims to explore **palmprint recognition** as a novel and more privacy-preserving biometric authentication method. Unlike traditional approaches such as fingerprint or facial recognition, palmprint-based verification offers increased security, resistance to spoofing, and better compliance with user privacy.

We leverage the powerful **ArcFace architecture** as the core of our recognition system and investigate the performance of various backbone networks including:

* **ResNet**
* **Vision Transformer (ViT)**

---

## 🧪 Experiment Setup

To configure and run experiments effectively, follow the steps below:

### 1. Dataset Configuration

Edit the dataset parameters in:

```bash
config/data
```

Specify dataset path, preprocessing settings, and splitting strategies.

### 2. Backbone Selection

Choose the model backbone (e.g., `resnet`, `vit`) in:

```bash
config/experiment
```

### 3. Model & Training Hyperparameters

Set up training-related parameters such as learning rate, batch size, loss function, etc., in:

```bash
config/model
```

---

## 🚀 Running Experiments

To start a training or evaluation run, execute the following command from the root directory:

```bash
bash run.sh
```

This script will automatically load configurations and launch the experiment based on your settings.
