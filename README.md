# Image Classification with CNN on CIFAR-10

**Deep Learning Capstone Project | Track A**

## English

### 1. Project Overview

This project trains a Convolutional Neural Network (CNN) to classify images from CIFAR-10 into 10 categories. The model is implemented in PyTorch and trained in Google Colab.

The experiments compare different training configurations. Validation accuracy is used to select the model, and the selected checkpoint is evaluated on the official CIFAR-10 test set.

### 2. Objectives

- Build an image classification model using PyTorch.
- Track training and validation performance.
- Compare training configurations, including dropout, learning rate, and weight decay.
- Compare the CNN with a majority-class baseline.
- Evaluate the selected model on the official test set.
- Examine classification errors and document the model's limitations.

### 3. Dataset

CIFAR-10 contains 60,000 color images, each measuring 32 × 32 pixels, across 10 classes.

| Split | Number of images |
|---|---:|
| Training set | 40,000 |
| Validation set | 10,000 |
| Official test set | 10,000 |
| **Total** | **60,000** |

The original 50,000-image training set was divided into training and validation subsets using a fixed random seed (`SEED = 0`). The official test set was kept separate during training and model selection.

The 10 classes are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

#### Preprocessing and Data Augmentation

- Convert images to PyTorch tensors.
- Normalize images using CIFAR-10 channel statistics.
- Apply random horizontal flips and random crops with padding during training.
- Apply normalization without random augmentation during validation and testing.

### 4. Model and Technologies

- **Model:** Custom Convolutional Neural Network (CNN)
- **Framework:** PyTorch
- **Language:** Python
- **Environment:** Google Colab
- **Hardware acceleration:** GPU when available
- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam

#### CNN Architecture

| Layer | Configuration |
|---|---|
| Convolutional layer 1 | 3 → 32 channels, 3 × 3 kernel, ReLU |
| Max pooling 1 | 2 × 2 |
| Convolutional layer 2 | 32 → 64 channels, 3 × 3 kernel, ReLU |
| Max pooling 2 | 2 × 2 |
| Flatten | 64 × 8 × 8 features |
| Fully connected layer | 4096 → 128, ReLU |
| Dropout | 0.25 |
| Output layer | 128 → 10 classes |

The model has approximately 545,098 trainable parameters.

#### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout probability | 0.25 |
| Epochs | 15 |
| Random seed | 0, set in the notebook |
| Model selection | Best validation accuracy |

### 5. Experimental Results and Evaluation

#### 5.1 Hyperparameter Experiments

The experiments examined different training-set sizes, dropout probabilities, learning rates, and weight-decay settings.

| Experiment | Configuration | Validation accuracy |
|---|---|---:|
| Initial baseline | 8,000 training images, 8 epochs, dropout 0.25, LR 0.001, WD 0.0001 | 56.50% |
| Dropout experiment | 8,000 training images, dropout 0.5, 8 epochs | 51.65% |
| Learning-rate experiment | 8,000 training images, LR 0.003, 8 epochs | 54.60% |
| Expanded baseline | 40,000 training images, 8 epochs | 70.31% |
| Dropout ablation | Dropout 0.0, 40,000 training images, 15 epochs | 75.15% |
| Weight-decay ablation | Weight decay 0.0, 40,000 training images, 15 epochs | 74.29% |
| Selected checkpoint run | Dropout 0.25, LR 0.001, WD 0.0001, 40,000 training images, 15 epochs | 72.83% |

The initial experiments used 8,000 training images. Later experiments used 40,000, so results across these groups cannot be used to isolate the effect of a single hyperparameter.

A separate 15-epoch run reached 76.17% validation accuracy at epoch 12. Its best-epoch weights were not saved, so that result was not used for the final test evaluation. A subsequent run produced the checkpoint used in this project, with a best validation accuracy of 72.83% at epoch 15.

The dropout and weight-decay experiments produced different validation results, but each configuration was tested in an individual run. Further controlled experiments would be needed to determine whether these differences are consistent.

#### 5.2 Final Test Evaluation

The selected checkpoint was evaluated on the official CIFAR-10 test set.

| Metric | Result |
|---|---:|
| Best validation accuracy | 72.83% |
| **Final test accuracy** | **72.49%** |
| Final test loss | 0.785 |
| Majority-class baseline accuracy | 10.00% |

The model achieved **72.49% test accuracy**, exceeding the project target of 65%. Its test accuracy was 0.34 percentage points lower than its best validation accuracy.

The majority-class baseline achieved 10.00% accuracy. CIFAR-10 has ten balanced classes, so always predicting one class provides a simple baseline for comparison.

#### 5.3 Per-Class Recall and Error Analysis

The following recall values were calculated on the validation set.

| Class | Validation recall |
|---|---:|
| Frog | 89.44% |
| Automobile | 89.13% |
| Ship | 84.74% |
| Horse | 82.61% |
| Truck | 80.81% |
| Airplane | 76.68% |
| Dog | 67.78% |
| Deer | 58.79% |
| Bird | 57.91% |
| Cat | 40.29% |

The model performed best on frog and automobile images. Cat had the lowest recall, followed by bird and deer.

The confusion matrix shows that cat images were frequently misclassified as dog (269 images) or frog (165 images). Deer images were sometimes classified as frog (125 images) or horse (109 images). These errors suggest that the model has difficulty distinguishing some classes with similar visual features.

The current CNN may benefit from additional convolutional layers or other changes to its feature extraction process.

**Note:** These per-class recall values and confusion-matrix observations come from the validation set, not the official test set.

### 6. Project Status

- [x] Create the GitHub repository.
- [x] Prepare and explore CIFAR-10.
- [x] Implement and train the baseline CNN.
- [x] Run hyperparameter experiments.
- [x] Select a checkpoint using validation accuracy.
- [x] Evaluate the selected checkpoint on the official test set.
- [x] Record evaluation results and analyze validation errors.
- [ ] Complete the final PDF report.
- [ ] Complete the project demonstration or video.
- [ ] Check that the repository contains all required files and instructions.

### 7. Reproducibility

The project was developed in Google Colab using Python, PyTorch, and torchvision.

#### Running the Project

1. Open the project notebook in Google Colab.
2. Run the dataset setup and preprocessing cells.
3. Train the CNN using the documented configuration.
4. Compare experiments using the validation set.
5. Load the selected checkpoint.
6. Evaluate the model on the official test set.

The CIFAR-10 dataset can be downloaded through `torchvision.datasets.CIFAR10`.

The notebook contains the model definition, data split, training procedure, evaluation functions, and plotting code. To reproduce an experiment, keep the data split, random seed, model configuration, and training settings consistent. GPU execution and random operations may still cause small differences between runs.

The `best_model.pth` file stores the selected model weights. It must be loaded into the matching CNN architecture before evaluation or inference.

The notebook and checkpoint should be made available in the repository or through documented download links. A file saved in a Colab runtime is not automatically accessible to other users.

### 8. Limitations and Future Work

- The CNN has a relatively simple architecture and struggles with some visually similar classes.
- Cat, bird, and deer have lower validation recall than the other classes.
- The initial and expanded experiments used different training-set sizes, limiting direct comparisons.
- Most configurations were tested only once.
- The model has not been compared with a deeper CNN architecture.

Future experiments could add batch normalization, use more convolutional blocks, apply learning-rate scheduling, and compare regularization settings under the same data split. Repeating experiments with multiple random seeds would also help establish whether the observed differences are consistent.

---

## ภาษาไทย

### 1. ภาพรวมโครงงาน

โครงงานนี้พัฒนาและฝึกสอนโมเดล Convolutional Neural Network (CNN) เพื่อจำแนกรูปภาพจากชุดข้อมูล CIFAR-10 ออกเป็น 10 ประเภท โดยใช้ PyTorch บน Google Colab

การทดลองศึกษาผลของการตั้งค่าการฝึกสอนที่แตกต่างกัน โดยใช้ Validation Accuracy ในการเลือกโมเดล และใช้ชุด Test อย่างเป็นทางการของ CIFAR-10 ในการประเมินผลขั้นสุดท้าย

### 2. วัตถุประสงค์

- สร้างโมเดลจำแนกรูปภาพด้วย PyTorch
- ติดตามผลการฝึกและผล Validation
- เปรียบเทียบการตั้งค่า Dropout, Learning Rate และ Weight Decay
- เปรียบเทียบ CNN กับ Majority-class Baseline
- ประเมินโมเดลที่เลือกด้วยชุด Test อย่างเป็นทางการ
- วิเคราะห์ข้อผิดพลาดและระบุข้อจำกัดของโมเดล

### 3. ชุดข้อมูล

CIFAR-10 ประกอบด้วยรูปภาพสี 60,000 ภาพ แต่ละภาพมีขนาด 32 × 32 พิกเซล แบ่งเป็น 10 คลาส

| ชุดข้อมูล | จำนวนภาพ |
|---|---:|
| Training set | 40,000 |
| Validation set | 10,000 |
| Official test set | 10,000 |
| **รวม** | **60,000** |

Training set เดิมจำนวน 50,000 ภาพถูกแบ่งเป็น Training และ Validation โดยใช้ Random Seed คงที่ (`SEED = 0`) ส่วน Test Set ถูกแยกออกจากการฝึกและการเลือกโมเดล

คลาสทั้ง 10 ประเภท ได้แก่

`airplane` (เครื่องบิน), `automobile` (รถยนต์), `bird` (นก), `cat` (แมว), `deer` (กวาง), `dog` (สุนัข), `frog` (กบ), `horse` (ม้า), `ship` (เรือ) และ `truck` (รถบรรทุก)

#### การเตรียมข้อมูลและ Data Augmentation

- แปลงรูปภาพเป็น PyTorch Tensor
- Normalize ภาพด้วยค่าสถิติของแต่ละช่องสีจาก CIFAR-10
- ใช้ Random Horizontal Flip ระหว่างการฝึก
- ใช้ Random Crop พร้อม Padding ระหว่างการฝึก
- ใช้ Normalize โดยไม่มี Random Augmentation กับชุด Validation และ Test

### 4. โมเดลและเทคโนโลยี

- **โมเดล:** Convolutional Neural Network (CNN) ที่ออกแบบขึ้นเอง
- **เฟรมเวิร์ก:** PyTorch
- **ภาษาโปรแกรม:** Python
- **สภาพแวดล้อม:** Google Colab
- **การเร่งความเร็ว:** GPU เมื่อพร้อมใช้งาน
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam

#### สถาปัตยกรรม CNN

| Layer | การตั้งค่า |
|---|---|
| Convolutional layer 1 | 3 → 32 Channels, Kernel 3 × 3, ReLU |
| Max pooling 1 | 2 × 2 |
| Convolutional layer 2 | 32 → 64 Channels, Kernel 3 × 3, ReLU |
| Max pooling 2 | 2 × 2 |
| Flatten | 64 × 8 × 8 Features |
| Fully connected layer | 4096 → 128, ReLU |
| Dropout | 0.25 |
| Output layer | 128 → 10 Classes |

โมเดลมีพารามิเตอร์ที่สามารถฝึกได้ประมาณ 545,098 ตัว

#### การตั้งค่าการฝึกสอน

| พารามิเตอร์ | ค่า |
|---|---|
| Batch size | 128 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout probability | 0.25 |
| จำนวน Epochs | 15 |
| Random seed | 0 ซึ่งกำหนดไว้ใน Notebook |
| เกณฑ์การเลือกโมเดล | Validation Accuracy สูงสุด |

### 5. ผลการทดลองและการประเมินผล

#### 5.1 การทดลอง Hyperparameters

การทดลองเปรียบเทียบขนาดชุด Training, Dropout, Learning Rate และ Weight Decay

| การทดลอง | การตั้งค่า | Validation Accuracy |
|---|---|---:|
| Baseline ชุดแรก | Training 8,000 ภาพ, 8 Epochs, Dropout 0.25, LR 0.001, WD 0.0001 | 56.50% |
| ทดลอง Dropout | Training 8,000 ภาพ, Dropout 0.5, 8 Epochs | 51.65% |
| ทดลอง Learning Rate | Training 8,000 ภาพ, LR 0.003, 8 Epochs | 54.60% |
| Baseline ชุดขยาย | Training 40,000 ภาพ, 8 Epochs | 70.31% |
| ทดลองตัด Dropout | Dropout 0.0, Training 40,000 ภาพ, 15 Epochs | 75.15% |
| ทดลองตัด Weight Decay | Weight decay 0.0, Training 40,000 ภาพ, 15 Epochs | 74.29% |
| รอบที่ใช้เลือก Checkpoint | Dropout 0.25, LR 0.001, WD 0.0001, Training 40,000 ภาพ, 15 Epochs | 72.83% |

การทดลองชุดแรกใช้ Training 8,000 ภาพ ส่วนการทดลองชุดหลังใช้ 40,000 ภาพ จึงไม่สามารถใช้เปรียบเทียบเพื่อสรุปผลของ Hyperparameter เพียงตัวเดียวได้โดยตรง

การฝึกอีกหนึ่งรอบจำนวน 15 Epochs เคยได้ Validation Accuracy 76.17% ใน Epoch 12 แต่ไม่ได้บันทึกน้ำหนักของ Epoch นั้นไว้ จึงไม่ได้นำมาใช้ประเมิน Test รอบสุดท้าย Checkpoint ที่เลือกใช้ในโครงงานนี้มาจากการฝึกรอบถัดมา ซึ่งได้ Validation Accuracy สูงสุด 72.83% ใน Epoch 15

ผลจากการทดลอง Dropout และ Weight Decay แตกต่างกัน แต่แต่ละการตั้งค่าทดลองเพียงรอบเดียว จึงควรทดลองเพิ่มเติมภายใต้เงื่อนไขเดียวกันก่อนสรุปว่าการตั้งค่าใดให้ผลดีกว่าอย่างสม่ำเสมอ

#### 5.2 ผลการประเมินด้วย Test Set

นำ Checkpoint ที่เลือกไปประเมินด้วยชุด Test อย่างเป็นทางการของ CIFAR-10

| ตัวชี้วัด | ผลลัพธ์ |
|---|---:|
| Best Validation Accuracy | 72.83% |
| **Final Test Accuracy** | **72.49%** |
| Final Test Loss | 0.785 |
| Majority-class Baseline Accuracy | 10.00% |

โมเดลได้ **Test Accuracy 72.49%** สูงกว่าเป้าหมายของโครงงานที่กำหนดไว้มากกว่า 65% โดยมีค่า Test Accuracy ต่ำกว่า Best Validation Accuracy อยู่ 0.34 จุดเปอร์เซ็นต์

Majority-class Baseline ได้ Accuracy 10.00% เนื่องจาก CIFAR-10 มี 10 คลาสที่มีจำนวนข้อมูลเท่ากัน จึงใช้การทำนายคลาสเดียวเป็นเกณฑ์เปรียบเทียบพื้นฐานได้

#### 5.3 Recall รายคลาสและการวิเคราะห์ข้อผิดพลาด

ค่า Recall ต่อไปนี้คำนวณจากชุด Validation

| คลาส | Validation Recall |
|---|---:|
| Frog (กบ) | 89.44% |
| Automobile (รถยนต์) | 89.13% |
| Ship (เรือ) | 84.74% |
| Horse (ม้า) | 82.61% |
| Truck (รถบรรทุก) | 80.81% |
| Airplane (เครื่องบิน) | 76.68% |
| Dog (สุนัข) | 67.78% |
| Deer (กวาง) | 58.79% |
| Bird (นก) | 57.91% |
| Cat (แมว) | 40.29% |

โมเดลทำได้ดีที่สุดในคลาส Frog และ Automobile ส่วน Cat มี Recall ต่ำที่สุด รองลงมาคือ Bird และ Deer

จาก Confusion Matrix พบว่าภาพ Cat ถูกจำแนกเป็น Dog จำนวน 269 ภาพ และ Frog จำนวน 165 ภาพ ส่วนภาพ Deer บางส่วนถูกจำแนกเป็น Frog จำนวน 125 ภาพ และ Horse จำนวน 109 ภาพ ผลนี้แสดงให้เห็นว่าโมเดลยังแยกบางคลาสที่มีลักษณะภาพคล้ายกันได้ไม่ดีนัก

การเพิ่ม Convolutional Layers หรือปรับปรุงการสกัดคุณลักษณะของภาพอาจช่วยแก้ปัญหานี้ได้

**หมายเหตุ:** ค่า Recall และตัวอย่างข้อผิดพลาดข้างต้นมาจากชุด Validation ไม่ใช่ชุด Test อย่างเป็นทางการ

### 6. สถานะโครงงาน

- [x] สร้าง GitHub Repository
- [x] เตรียมและสำรวจชุดข้อมูล CIFAR-10
- [x] สร้างและฝึกสอน CNN Baseline
- [x] ทดลองปรับ Hyperparameters
- [x] เลือก Checkpoint ด้วย Validation Accuracy
- [x] ประเมิน Checkpoint ด้วยชุด Test อย่างเป็นทางการ
- [x] บันทึกผลการประเมินและวิเคราะห์ข้อผิดพลาดจาก Validation
- [ ] จัดทำรายงาน PDF ฉบับสมบูรณ์
- [ ] จัดทำการสาธิตหรือวิดีโอโครงงาน
- [ ] ตรวจสอบไฟล์และคำแนะนำการใช้งานใน Repository

### 7. การทำซ้ำการทดลอง

โครงงานพัฒนาด้วย Python, PyTorch และ torchvision บน Google Colab

#### ขั้นตอนการใช้งาน

1. เปิด Notebook ของโครงงานใน Google Colab
2. รันเซลล์สำหรับเตรียมชุดข้อมูลและ Preprocessing
3. ฝึกสอน CNN ตามการตั้งค่าที่ระบุ
4. เปรียบเทียบผลการทดลองด้วยชุด Validation
5. โหลด Checkpoint ที่เลือก
6. ประเมินโมเดลด้วยชุด Test อย่างเป็นทางการ

สามารถดาวน์โหลด CIFAR-10 ผ่าน `torchvision.datasets.CIFAR10` ได้โดยอัตโนมัติ

Notebook ประกอบด้วยโค้ดประกาศโมเดล การแบ่งข้อมูล การฝึกสอน ฟังก์ชันประเมินผล และการสร้างกราฟ หากต้องการทำซ้ำการทดลอง ควรใช้ Data Split, Random Seed, Model Configuration และ Training Settings เดียวกัน อย่างไรก็ตาม การทำงานบน GPU และการสุ่มบางส่วนอาจทำให้ผลต่างกันเล็กน้อยระหว่างรอบ

ไฟล์ `best_model.pth` เก็บน้ำหนักของโมเดลที่เลือก ต้องโหลดเข้ากับ CNN Architecture ที่ตรงกันก่อนนำไปประเมินผลหรือทำนายภาพ

ควรตรวจสอบว่า Notebook และ Checkpoint พร้อมใช้งานใน Repository หรือมีลิงก์ดาวน์โหลดที่ระบุไว้ชัดเจน เพราะไฟล์ที่บันทึกใน Colab Runtime จะไม่สามารถเข้าถึงได้โดยผู้ใช้อื่นโดยอัตโนมัติ

### 8. ข้อจำกัดและแนวทางพัฒนาต่อ

- CNN มีสถาปัตยกรรมค่อนข้างเรียบง่ายและยังจำแนกบางคลาสที่คล้ายกันได้ไม่ดี
- Cat, Bird และ Deer มี Validation Recall ต่ำกว่าคลาสอื่น
- การทดลองชุดแรกและชุดขยายใช้จำนวนภาพ Training ต่างกัน จึงเปรียบเทียบผลโดยตรงได้จำกัด
- การตั้งค่าส่วนใหญ่ทดลองเพียงรอบเดียว
- ยังไม่ได้เปรียบเทียบกับ CNN ที่มีสถาปัตยกรรมลึกกว่า

การทดลองในอนาคตอาจเพิ่ม Batch Normalization และ Convolutional Blocks ใช้ Learning-rate Scheduling เปรียบเทียบ Regularization ภายใต้ Data Split เดียวกัน และทดลองซ้ำด้วยหลาย Random Seeds เพื่อดูว่าผลที่ได้มีความสม่ำเสมอหรือไม่

---

**Author:** AkanoFah  
**Project:** Deep Learning Capstone: Image Classification with CNN on CIFAR-10
