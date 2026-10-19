# Image Classification with CNN on CIFAR-10

**Deep Learning Capstone Project**

## English

### 1. Project Overview

This project builds and evaluates a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model classifies 32 × 32 color images into 10 categories.

The project covers data preprocessing, CNN implementation, model training, hyperparameter experiments, and evaluation on unseen test data.

### 2. Project Objectives

- Build a CNN for multi-class image classification.
- Apply image preprocessing and data augmentation.
- Train and evaluate the model using PyTorch.
- Compare model configurations by changing regularization settings.
- Evaluate the final model against a majority-class baseline.

### 3. Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 color images across 10 classes.

| Dataset split | Number of images |
|---|---:|
| Training | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |
| **Total** | **60,000** |

The original 50,000-image training set is split into 40,000 training images and 10,000 validation images. The official 10,000-image test set is reserved for final evaluation.

The 10 classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### 4. Data Preprocessing

All images are converted to tensors and normalized. Training images also use data augmentation to improve generalization.

**Training transformations**
- Random horizontal flip
- Random crop to 32 × 32 pixels with padding of 4 pixels
- Tensor conversion
- Normalization

**Validation and test transformations**
- Tensor conversion
- Normalization

Validation and test images do not use random augmentation.

### 5. Model Architecture

The CNN contains two convolutional blocks followed by fully connected layers.

| Layer | Configuration |
|---|---|
| Convolution 1 | 3 input channels → 32 output channels, kernel size 3 |
| Activation | ReLU |
| Pooling 1 | Max pooling, 2 × 2 |
| Convolution 2 | 32 input channels → 64 output channels, kernel size 3 |
| Activation | ReLU |
| Pooling 2 | Max pooling, 2 × 2 |
| Flatten | 64 × 8 × 8 features |
| Fully connected 1 | 4096 → 128 |
| Activation | ReLU |
| Dropout | Configurable dropout probability |
| Output layer | 128 → 10 classes |

The model has **545,098 trainable parameters**.

### 6. Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Google Colab
- CIFAR-10

### 7. Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 0 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Epochs | 15 |
| Baseline dropout | 0.25 |
| Baseline weight decay | 0.0001 |

The baseline and experimental models use the same dataset split, batch size, learning rate, optimizer, and number of epochs to make their validation results easier to compare.

### 8. Experimental Results

Three model configurations were compared to examine the effects of weight decay and dropout.

| Experiment | Dropout | Weight decay | Train accuracy | Validation accuracy | Train–validation gap |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.25 | 0.0001 | 72.77% | 73.97% | -1.20 pp |
| Weight decay removed | 0.25 | 0.0 | 72.66% | 73.81% | -1.15 pp |
| Dropout removed | 0.0 | 0.0001 | 74.55% | 75.25% | -0.70 pp |

*Train–validation gap = training accuracy − validation accuracy. Values are reported in percentage points (pp).*

The model without dropout achieved the highest validation accuracy among the tested configurations, reaching **75.25%**. Removing weight decay produced a slightly lower validation accuracy than the baseline.

These results apply to the configurations tested in this project; they do not establish that removing dropout will improve every CNN.

### 9. Final Test Evaluation

The model selected using validation performance was evaluated on the official CIFAR-10 test set.

| Metric | Result |
|---|---:|
| Test loss | 0.736 |
| Test accuracy | 74.73% |
| Majority-class baseline accuracy | 10.00% |
| Target test accuracy | Above 65% |
| Target achieved | Yes |

The final model achieved **74.73% test accuracy**, exceeding the project's target of 65%.

The majority-class baseline predicts the same most frequent class for every image. Since CIFAR-10 has 10 equally sized classes, this baseline achieves 10% accuracy.

### 10. Saved Model Files

The following files can be saved after training:

| File | Description |
|---|---|
| `best_model.pth` | Trained model weights |
| `best_model_config.json` | Model configuration and recorded experiment metrics |

Save the selected model weights with:

```python
torch.save(model_exp2.state_dict(), "best_model.pth")
```

The configuration file can be saved separately as JSON. These files must be uploaded to the repository or distributed separately if others need to download them. Their presence in the local Colab session does not automatically add them to GitHub.

To load the saved weights, first recreate the same CNN architecture with the same configuration:

```python
model = CNN(n_classes=10, p_drop=0.0)
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)
model = model.to(device)
model.eval()
```

Here, `device` should be set to the available execution device, such as a GPU or CPU.

## 11. Installation & Reproduction

### Requirements

- Python 3.10 or later
- Git
- Dependencies listed in `requirements.txt`

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AkanoFah/dl-image-classification.git
cd dl-image-classification
python -m pip install -r requirements.txt
```

### Run the Notebook

Open `cifar10_cnn.ipynb` in Google Colab or Jupyter Notebook and run the cells in order.

The notebook covers dataset loading, preprocessing, model training, validation, learning curves, confusion matrix analysis, and final test evaluation.

### Training Configuration

- Random seed: `0`
- Batch size: `128`
- Epochs: `15`
- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: Cross-Entropy Loss

The CIFAR-10 dataset is downloaded automatically through `torchvision` when required. Use the existing training, validation, and test split to reproduce the experiment.

For reliable evaluation, do not use test-set results to select or tune hyperparameters. Results may vary slightly depending on software versions and hardware.

### Model Weights

The trained model weights (`best_model.pth`) and configuration file (`best_model_config.json`) are not included in this repository. Run the notebook to train the model and generate the files when needed.

The selected model uses a CNN with 10 output classes and a dropout probability of `0.0`. To load saved weights, recreate the same model architecture and configuration before loading the state dictionary.

---

## ภาษาไทย

### 1. ภาพรวมโครงงาน

โครงงานนี้เป็นการสร้างและประเมินโมเดล Convolutional Neural Network (CNN) สำหรับจำแนกภาพจากชุดข้อมูล CIFAR-10 โดยโมเดลจะจำแนกภาพสีขนาด 32 × 32 พิกเซล ออกเป็น 10 คลาส

เนื้อหาครอบคลุมการเตรียมข้อมูล การสร้าง CNN การฝึกโมเดล การทดลองปรับค่าพารามิเตอร์ และการประเมินผลด้วยชุดข้อมูลทดสอบ

### 2. วัตถุประสงค์

- สร้างโมเดล CNN สำหรับจำแนกภาพหลายคลาส
- เรียนรู้การเตรียมข้อมูลและการทำ Data Augmentation
- ฝึกและประเมินโมเดลด้วย PyTorch
- เปรียบเทียบโมเดลโดยปรับการตั้งค่าด้าน Regularization
- ประเมินโมเดลสุดท้ายเทียบกับ Majority-class Baseline

### 3. ชุดข้อมูล

โครงงานใช้ชุดข้อมูล CIFAR-10 ซึ่งประกอบด้วยภาพสีทั้งหมด 60,000 ภาพ แบ่งเป็น 10 คลาส

| ชุดข้อมูล | จำนวนภาพ |
|---|---:|
| Training | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |
| **รวม** | **60,000** |

ข้อมูล Training เดิมจำนวน 50,000 ภาพถูกแบ่งเป็น Training 40,000 ภาพ และ Validation 10,000 ภาพ ส่วน Test ใช้ชุดข้อมูลทดสอบอย่างเป็นทางการจำนวน 10,000 ภาพ เพื่อประเมินผลขั้นสุดท้าย

คลาสทั้ง 10 ได้แก่

- เครื่องบิน (Airplane)
- รถยนต์ (Automobile)
- นก (Bird)
- แมว (Cat)
- กวาง (Deer)
- สุนัข (Dog)
- กบ (Frog)
- ม้า (Horse)
- เรือ (Ship)
- รถบรรทุก (Truck)

### 4. การเตรียมข้อมูล

ภาพทั้งหมดถูกแปลงเป็น Tensor และทำ Normalization ส่วนข้อมูล Training จะผ่าน Data Augmentation เพิ่มเติม เพื่อช่วยให้โมเดลเรียนรู้จากรูปแบบภาพที่หลากหลายขึ้น

**การแปลงข้อมูล Training**
- สุ่มพลิกภาพในแนวนอน
- สุ่ม Crop ภาพขนาด 32 × 32 พิกเซล โดยเพิ่ม Padding 4 พิกเซล
- แปลงภาพเป็น Tensor
- ทำ Normalization

**การแปลงข้อมูล Validation และ Test**
- แปลงภาพเป็น Tensor
- ทำ Normalization

ชุดข้อมูล Validation และ Test จะไม่ใช้ Random Augmentation เพื่อให้การประเมินผลไม่ขึ้นอยู่กับการสุ่มแปลงภาพ

### 5. สถาปัตยกรรมโมเดล

โมเดล CNN ประกอบด้วย Convolutional Block จำนวน 2 บล็อก ตามด้วย Fully Connected Layer

| Layer | การตั้งค่า |
|---|---|
| Convolution 1 | 3 ช่องสัญญาณเข้า → 32 ช่องสัญญาณออก, Kernel size 3 |
| Activation | ReLU |
| Pooling 1 | Max Pooling ขนาด 2 × 2 |
| Convolution 2 | 32 ช่องสัญญาณเข้า → 64 ช่องสัญญาณออก, Kernel size 3 |
| Activation | ReLU |
| Pooling 2 | Max Pooling ขนาด 2 × 2 |
| Flatten | 64 × 8 × 8 Features |
| Fully Connected 1 | 4096 → 128 |
| Activation | ReLU |
| Dropout | ปรับค่า Dropout ได้ |
| Output Layer | 128 → 10 คลาส |

โมเดลมีพารามิเตอร์ที่สามารถฝึกได้ทั้งหมด **545,098 พารามิเตอร์**

### 6. เทคโนโลยีที่ใช้

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Google Colab
- CIFAR-10

### 7. การตั้งค่าการฝึกโมเดล

| พารามิเตอร์ | ค่า |
|---|---|
| Random seed | 0 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| จำนวน Epochs | 15 |
| Dropout ของ Baseline | 0.25 |
| Weight decay ของ Baseline | 0.0001 |

โมเดล Baseline และโมเดลที่ใช้ทดลองจะใช้การแบ่งชุดข้อมูล Batch size, Learning rate, Optimizer และจำนวน Epochs เดียวกัน เพื่อให้เปรียบเทียบผล Validation ได้ง่ายขึ้น

### 8. ผลการทดลอง

เปรียบเทียบการตั้งค่าโมเดล 3 รูปแบบ
เพื่อศึกษาผลของ Weight Decay และ Dropout

| การทดลอง | Dropout | Weight decay | Train accuracy | Validation accuracy | Train–validation gap |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.25 | 0.0001 | 72.77% | 73.97% | -1.20 pp |
| ปิด Weight decay | 0.25 | 0.0 | 72.66% | 73.81% | -1.15 pp |
| ปิด Dropout | 0.0 | 0.0001 | 74.55% | 75.25% | -0.70 pp |

*Train–validation gap คำนวณจาก Training accuracy − Validation accuracy โดย pp หมายถึง Percentage points*

โมเดลที่ปิด Dropout ได้ Validation accuracy สูงที่สุดในบรรดาการทดลองที่ทำ โดยมีค่า **75.25%** ส่วนการปิด Weight Decay ให้ผล Validation accuracy ต่ำกว่า Baseline เล็กน้อย

ผลนี้สรุปได้เฉพาะการตั้งค่าที่ทดลองในโครงงานนี้ ไม่ได้หมายความว่าการปิด Dropout จะช่วยให้ CNN ทุกโมเดลมีประสิทธิภาพดีขึ้น

### 9. ผลการประเมินบนชุดข้อมูลทดสอบ

โมเดลที่เลือกจากผล Validation ถูกนำไปประเมินด้วยชุดข้อมูล Test อย่างเป็นทางการของ CIFAR-10

| ตัวชี้วัด | ผลลัพธ์ |
|---|---:|
| Test loss | 0.736 |
| Test accuracy | 74.73% |
| Majority-class baseline accuracy | 10.00% |
| เป้าหมาย Test accuracy | มากกว่า 65% |
| ผ่านเป้าหมาย | ผ่าน |

โมเดลสุดท้ายมี **Test accuracy เท่ากับ 74.73%** ซึ่งสูงกว่าเป้าหมายที่กำหนดไว้ที่ 65%

Majority-class baseline เป็นวิธีพื้นฐานที่ทำนายทุกภาพให้เป็นคลาสเดียวกัน เนื่องจาก CIFAR-10 มีจำนวนภาพเท่ากันในแต่ละคลาส วิธีนี้จึงมี Accuracy เท่ากับ 10%

### 10. ไฟล์โมเดลที่บันทึกไว้

หลังจากฝึกโมเดลแล้ว สามารถบันทึกไฟล์ต่อไปนี้ได้

| ไฟล์ | รายละเอียด |
|---|---|
| `best_model.pth` | น้ำหนักของโมเดลที่ฝึกแล้ว |
| `best_model_config.json` | การตั้งค่าโมเดลและค่าผลการทดลองที่บันทึกไว้ |

คำสั่งบันทึกน้ำหนักโมเดล:

```python
torch.save(model_exp2.state_dict(), "best_model.pth")
```

ไฟล์ Configuration สามารถบันทึกแยกเป็น JSON ได้ หากต้องการให้ผู้อื่นดาวน์โหลดไฟล์เหล่านี้ ต้องอัปโหลดไปยัง GitHub หรือส่งแยกต่างหาก การบันทึกไฟล์ไว้ใน Colab ไม่ได้ทำให้ไฟล์ปรากฏใน GitHub โดยอัตโนมัติ

เมื่อต้องการโหลดโมเดล ให้สร้าง CNN ที่มีสถาปัตยกรรมและการตั้งค่าเดียวกับโมเดลเดิมก่อน แล้วจึงโหลดน้ำหนัก:

```python
model = CNN(n_classes=10, p_drop=0.0)
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)
model = model.to(device)
model.eval()
```

ตัวแปร `device` ต้องกำหนดให้เป็นอุปกรณ์ที่ใช้รัน เช่น GPU หรือ CPU

## 11. การติดตั้งและการทำซ้ำการทดลอง

### สิ่งที่ต้องมี

- Python เวอร์ชัน 3.10 ขึ้นไป
- Git
- ไลบรารีที่ระบุไว้ในไฟล์ `requirements.txt`

### การติดตั้ง

โคลน Repository และติดตั้งไลบรารีที่จำเป็นด้วยคำสั่งต่อไปนี้

```bash
git clone https://github.com/AkanoFah/dl-image-classification.git
cd dl-image-classification
python -m pip install -r requirements.txt
```

### วิธีเรียกใช้งาน Notebook

เปิดไฟล์ `cifar10_cnn.ipynb` ด้วย Google Colab หรือ Jupyter Notebook แล้วเรียกใช้งานเซลล์ตามลำดับ

Notebook ครอบคลุมการโหลดและเตรียมข้อมูล การฝึกโมเดล การประเมินผล Validation การแสดง Learning Curves และ Confusion Matrix รวมถึงการประเมินผลขั้นสุดท้ายบนชุดข้อมูล Test

### การตั้งค่าการฝึกโมเดล

- Random Seed: `0`
- Batch Size: `128`
- จำนวน Epochs: `15`
- Optimizer: Adam
- Learning Rate: `0.001`
- Loss Function: Cross-Entropy Loss

ระบบจะดาวน์โหลดชุดข้อมูล CIFAR-10 ผ่าน `torchvision` โดยอัตโนมัติเมื่อจำเป็น ควรใช้การแบ่งชุดข้อมูล Training, Validation และ Test ตามที่กำหนดไว้ เพื่อให้สามารถทำซ้ำการทดลองได้อย่างเหมาะสม

ห้ามใช้ผลจากชุดข้อมูล Test ในการเลือกหรือปรับแต่ง Hyperparameters เพื่อป้องกันข้อมูลรั่วไหลในการประเมินผล ทั้งนี้ ผลลัพธ์อาจแตกต่างกันเล็กน้อยตามเวอร์ชันซอฟต์แวร์และอุปกรณ์ที่ใช้ประมวลผล

### ไฟล์น้ำหนักโมเดล

Repository นี้ **ไม่ได้รวมไฟล์น้ำหนักโมเดล `best_model.pth` และไฟล์การตั้งค่า `best_model_config.json`** ผู้ใช้ต้องเรียกใช้งาน Notebook เพื่อฝึกโมเดลและสร้างไฟล์ดังกล่าวเมื่อต้องการใช้งาน

โมเดลที่เลือกใช้มีเอาต์พุต 10 คลาส และกำหนดค่า Dropout เป็น `0.0` หากต้องการโหลดน้ำหนักโมเดลที่บันทึกไว้ ต้องสร้างสถาปัตยกรรมและการตั้งค่าของโมเดลให้ตรงกับโมเดลเดิมก่อน
