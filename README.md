# Adversarial Attack and Defense

End-to-end adversarial attack, analysis, restoration, and defense pipeline on a filtered subset of the NaturalImageNet dataset. The project starts from baseline evaluation with a pretrained MobileNetV2, attacks the best-performing class with two FGSM variants, analyzes perturbation fingerprints in both pixel and frequency domains, applies restoration and spectral denoising, and finally fine-tunes a classifier on a mixed clean/adversarial/restored dataset.

## Overview

This repository implements a complete experimental workflow:

1. **Filter the dataset** into a balanced subset.
2. **Evaluate baseline accuracy** per class with pretrained MobileNetV2.
3. **Attack the top-performing class** (`king_penguin`) with two FGSM variants.
4. **Compute perturbation fingerprints** using mean and median aggregation.
5. **Attempt restoration** by subtracting statistical fingerprints.
6. **Analyze adversarial spectra** in the Fourier domain.
7. **Apply spectral denoising** with radius-based low-pass filtering.
8. **Create a larger mixed dataset** containing clean, attacked, restored, and denoised images.
9. **Fine-tune MobileNetV2** on the mixed dataset for improved robustness.

---

## Key Results

### Baseline class accuracy

| Class | Accuracy (%) |
|---|---:|
| African elephant | 73.00 |
| brown bear | 89.00 |
| chameleon | 71.00 |
| dragonfly | 82.00 |
| giant panda | 96.33 |
| gorilla | 90.00 |
| king penguin | 97.33 |
| koala | 96.00 |
| ladybug | 93.00 |
| lion | 94.00 |
| meerkat | 95.00 |
| orangutan | 93.67 |
| red fox | 55.33 |
| snail | 86.67 |
| tiger | 96.67 |
| kite | 78.33 |
| Virginia deer | 44.00 |

**Top class selected for attack:** `king_penguin`

### Attack effectiveness

| Attack | Epsilon | Accuracy After Attack (%) |
|---|---:|---:|
| FGSM1 | 0.30 | 51.00 |
| FGSM2 (de-norm → attack → re-norm) | 0.10 | 21.33 |

### Fingerprint-based restoration

| Method | FGSM1 Accuracy (%) | FGSM2 Accuracy (%) |
|---|---:|---:|
| Adversarial | 51.00 | 21.33 |
| Restored (Mean fingerprint) | 50.67 | 21.33 |
| Restored (Median fingerprint) | 50.33 | 21.67 |

### Spectral denoising

| Method | FGSM1 Accuracy (%) | FGSM2 Accuracy (%) |
|---|---:|---:|
| Original | 97.00 | 97.00 |
| Adversarial | 51.00 | 21.33 |
| Spectrally restored (`radius = 90`) | 69.67 | 48.00 |

### Fine-tuning on the mixed dataset

| Metric | Value |
|---|---:|
| Initial validation accuracy | 30.09% |
| Final test accuracy | 99.40% |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | StepLR(step=7, gamma=0.1) |
| Total training time | ~1507 s (~25 min) |



## Methodology

### 1) Dataset filtering

The project begins by filtering the original NaturalImageNet directory into a smaller balanced subset. The filtering script copies up to **300 images per class** into `Filtered_Dataset`, making the downstream evaluation more uniform and easier to analyze.

### 2) Baseline evaluation

`InitialDetection.py` evaluates a pretrained **MobileNetV2** on the filtered subset using ImageNet normalization. It maps the folder names to the corresponding ImageNet class indices and reports per-class accuracy. The best-performing class is then chosen as the primary target for attack analysis.

### 3) Two FGSM attack variants

The repository studies two single-step gradient attacks on the selected class:

| Variant | Description |
|---|---|
| **FGSM1** | Adds `epsilon * sign(gradient)` directly in the normalized input space. |
| **FGSM2** | De-normalizes the image, applies FGSM in pixel space, then normalizes again before inference. |

FGSM2 is much stronger in terms of accuracy degradation, while FGSM1 tends to produce less visually obvious perturbations.

### 4) Statistical fingerprints

After generating adversarial examples, the project computes perturbation fingerprints:

- **Average fingerprint**: pixel-wise mean perturbation.
- **Median fingerprint**: pixel-wise median perturbation.

These summarize common perturbation patterns across all attacked examples of the target class.

### 5) Fingerprint subtraction restoration

The next step subtracts the learned mean or median fingerprint from attacked images. This is a simple statistical recovery attempt. In this project, it does **not** recover much accuracy, which suggests that adversarial perturbations are not well represented by a single universal additive template.

### 6) Frequency-domain analysis

The project then converts clean images, adversarial images, and perturbations to the frequency domain using **2D FFT**. The spectra show that clean and adversarial images remain globally similar, while perturbations concentrate more strongly in higher-frequency regions, with some low-frequency leakage.

### 7) Spectral denoising

Using that spectral observation, the project applies a **radius-based low-pass filter** in the Fourier domain. Recovery improves substantially as the radius increases, peaks around **radius = 90**, and then starts to decline as more high-frequency content is reintroduced.

### 8) Fine-tuning for robustness

Finally, the project creates a mixed multiclass dataset containing:

- original clean images
- FGSM1 images
- FGSM2 images
- mean-restored images
- median-restored images
- spectrally denoised images

A pretrained MobileNetV2 is then fine-tuned on this dataset and reaches **99.40% final test accuracy**.

---

## Dataset Layout

### Filtered dataset

```text
Filtered_Dataset/
├── African elephant/
├── brown bear/
├── chameleon/
├── ...
└── king penguin/
```

### Mixed dataset created for fine-tuning

```text
Dataset/
├── Filtered Dataset/
│   └── <class folders>
├── Adversarial Dataset/
│   ├── FGSM1/
│   │   └── <class folders>
│   └── FGSM2/
│       └── <class folders>
├── Restored/
│   ├── FGSM1/
│   │   ├── avg/
│   │   └── median/
│   └── FGSM2/
│       ├── avg/
│       └── median/
└── Denoised/
    ├── FGSM1/
    │   └── radius 90/
    └── FGSM2/
        └── radius 90/
```

### Dataset scale used for fine-tuning

| Quantity | Value |
|---|---:|
| Number of classes | 17 |
| Images per class | 300 |
| Stages/folders per class | 9 |
| Total images | 45,900 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 (pretrained) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Batch size | `64` |
| Epochs | `10` |
| Scheduler | StepLR |
| Scheduler step size | `7` |
| Scheduler gamma | `0.1` |
| Train / Val / Test split | `70% / 15% / 15%` |
| DataLoader workers | `14` |

### Data augmentation

| Augmentation | Setting |
|---|---|
| RandomResizedCrop | `224`, scale=`(0.8, 1.0)` |
| RandomHorizontalFlip | Enabled |
| ColorJitter | brightness=`0.2`, contrast=`0.2`, saturation=`0.2`, hue=`0.1` |
| RandomRotation | `15°` |

### System configuration

| Component | Value |
|---|---|
| GPU | Nvidia RTX 4070m (8GB) |
| CPU | AMD Ryzen 7 8845HS |
| RAM | 32 GB |
| Framework | PyTorch 2.1.0 |
| CUDA | 11.8 |
| OS | Windows 11 Home 24H2 |

---

## Figures

### Attack accuracy vs epsilon

![Model Accuracy After FGSM Attack on King Penguins vs Epsilon](/images/Figure_1.png)

### Frequency spectrum analysis

<table>
  <tr>
    <td align="center">
      <img src="/images/Figure_2.png" alt="Average spectrum analysis for FGSM1" width="100%"><br>
      <sub>FGSM1 spectrum analysis</sub>
    </td>
    <td align="center">
      <img src="/images/Figure_3.png" alt="Average spectrum analysis for FGSM2" width="100%"><br>
      <sub>FGSM2 spectrum analysis</sub>
    </td>
  </tr>
</table>

### Radius-wise spectral denoising

![Model Accuracy on Both Denoised FGSM vs Low-pass Filter Radius](/images/Figure_4.png)

### Fine-tuning curves

<table>
  <tr>
    <td align="center">
      <img src="/images/Figure_5.png" alt="Accuracy vs epochs" width="100%"><br>
      <sub>Accuracy vs epochs</sub>
    </td>
    <td align="center">
      <img src="/images/Figure_6.png" alt="Loss vs epochs" width="100%"><br>
      <sub>Loss vs epochs</sub>
    </td>
  </tr>
</table>

### Visual examples

![FGSM1 denoised images](/images/Figure%206.2.png)

---

## Main Takeaways

| Finding | Outcome |
|---|---|
| Baseline model is strong on `king_penguin` | 97.33% baseline accuracy |
| FGSM2 is stronger than FGSM1 | Larger accuracy drop at smaller epsilon |
| Mean/median fingerprint subtraction is weak | Little to no recovery |
| Frequency-domain filtering helps | Best recovery around radius 90 |
| Mixed-data fine-tuning is the strongest defense here | 99.40% final test accuracy |

---
