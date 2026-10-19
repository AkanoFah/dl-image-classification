# Image Classification with CNN on CIFAR-10

**Deep Learning Capstone Project**

## English

### 1. Project Overview

This project uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories. The model will be built and trained with PyTorch, then evaluated on images that were not used during training.

The project also explores how training configurations affect classification performance.

### 2. Objectives

- Build an image classification model using PyTorch.
- Train a CNN and monitor its performance on a validation set.
- Experiment with hyperparameters to find effective training configurations.
- Compare the CNN against a baseline model.
- Evaluate the selected model on the official CIFAR-10 test set.
- Record the experimental results, limitations, and steps needed to reproduce the experiments.

### 3. Dataset

CIFAR-10 contains 60,000 color images, each measuring 32 × 32 pixels, across 10 classes.

| Split | Number of images |
|---|---:|
| Training set | 50,000 |
| Test set | 10,000 |
| Total | 60,000 |

The training set will be divided into training and validation subsets. The official test set will remain separate until the final evaluation to avoid using test results to guide model selection.

The 10 classes are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

### 4. Model and Technologies

- **Model:** Convolutional Neural Network (CNN)
- **Framework:** PyTorch
- **Language:** Python
- **Development environment:** Google Colab
- **Hardware acceleration:** GPU when available

The CNN architecture, preprocessing steps, training settings, and selected hyperparameters will be documented alongside the implementation.

### 5. Evaluation

The main evaluation metric is classification accuracy. Additional metrics, such as precision, recall, F1-score, and a confusion matrix, may be included to examine performance across individual classes.

Validation results will be used to compare training configurations and select a model. The official test set will be used for the final evaluation.

Results will be reported after the experiments have been completed. No performance figures are claimed at this stage.

### 6. Project Status

- [x] Create the GitHub repository.
- [ ] Prepare and explore the dataset.
- [ ] Train the baseline CNN.
- [ ] Run hyperparameter experiments.
- [ ] Evaluate the selected model on the test set.
- [ ] Document results and limitations.
- [ ] Complete the project report and demonstration.

### 7. Reproducibility

Setup instructions, dependencies, and commands for training and evaluation will be added once the project structure and scripts are finalized.

The final documentation will include the model configuration and relevant training settings so that the experiments can be repeated.

---

## ภาษาไทย

### 1. ภาพรวมโครงงาน

โครงงานนี้พัฒนาโมเดล Convolutional Neural Network (CNN) เพื่อจำแนกรูปภาพจากชุดข้อมูล CIFAR-10 ออกเป็น 10 ประเภท โดยใช้ PyTorch ในการสร้างและฝึกสอนโมเดล จากนั้นประเมินผลด้วยรูปภาพที่ไม่ได้ใช้ในการฝึกสอน

นอกจากนี้ ยังทดลองปรับค่าการฝึกสอนเพื่อศึกษาว่าการตั้งค่าต่าง ๆ ส่งผลต่อประสิทธิภาพการจำแนกรูปภาพอย่างไร

### 2. วัตถุประสงค์

- สร้างโมเดลจำแนกรูปภาพด้วย PyTorch
- ฝึกสอน CNN และติดตามผลด้วยชุด Validation
- ทดลองปรับ Hyperparameters เพื่อหาการตั้งค่าที่เหมาะสม
- เปรียบเทียบ CNN กับโมเดล Baseline
- ประเมินโมเดลที่เลือกด้วยชุด Test อย่างเป็นทางการของ CIFAR-10
- บันทึกผลการทดลอง ข้อจำกัด และขั้นตอนที่จำเป็นสำหรับการทำซ้ำ

### 3. ชุดข้อมูล

CIFAR-10 ประกอบด้วยรูปภาพสีจำนวน 60,000 ภาพ แต่ละภาพมีขนาด 32 × 32 พิกเซล แบ่งเป็น 10 ประเภท

| ชุดข้อมูล | จำนวนภาพ |
|---|---:|
| Training set | 50,000 |
| Test set | 10,000 |
| รวม | 60,000 |

จะแบ่ง Training set ออกเป็นชุด Training และ Validation ส่วน Test set อย่างเป็นทางการจะแยกไว้จนถึงขั้นตอนประเมินผลสุดท้าย เพื่อไม่ให้ผลจากชุดทดสอบมีอิทธิพลต่อการเลือกโมเดล

รูปภาพทั้ง 10 ประเภท ได้แก่

`airplane` (เครื่องบิน), `automobile` (รถยนต์), `bird` (นก), `cat` (แมว), `deer` (กวาง), `dog` (สุนัข), `frog` (กบ), `horse` (ม้า), `ship` (เรือ) และ `truck` (รถบรรทุก)

### 4. โมเดลและเทคโนโลยี

- **โมเดล:** Convolutional Neural Network (CNN)
- **เฟรมเวิร์ก:** PyTorch
- **ภาษาโปรแกรม:** Python
- **สภาพแวดล้อมการพัฒนา:** Google Colab
- **การเร่งความเร็วด้วยฮาร์ดแวร์:** GPU เมื่อพร้อมใช้งาน

จะบันทึกรายละเอียดสถาปัตยกรรม CNN ขั้นตอนเตรียมข้อมูล การตั้งค่าการฝึกสอน และ Hyperparameters ที่เลือกไว้ควบคู่กับโค้ด

### 5. การประเมินผล

ใช้ Classification Accuracy เป็นตัวชี้วัดหลัก และอาจเพิ่ม Precision, Recall, F1-score และ Confusion Matrix เพื่อวิเคราะห์ผลการจำแนกในแต่ละประเภท

ผลจากชุด Validation จะใช้เปรียบเทียบการตั้งค่าการฝึกสอนและเลือกโมเดล ส่วนชุด Test จะใช้สำหรับการประเมินผลขั้นสุดท้าย

จะรายงานผลเมื่อการทดลองเสร็จสิ้น โดยในขั้นตอนนี้ยังไม่มีการระบุตัวเลขประสิทธิภาพของโมเดล

### 6. สถานะโครงงาน

- [x] สร้าง GitHub Repository
- [ ] เตรียมและสำรวจชุดข้อมูล
- [ ] ฝึกสอน CNN Baseline
- [ ] ทดลองปรับ Hyperparameters
- [ ] ประเมินโมเดลที่เลือกด้วยชุด Test
- [ ] บันทึกผลการทดลองและข้อจำกัด
- [ ] จัดทำรายงานและการสาธิตโครงงาน

### 7. การทำซ้ำการทดลอง

จะเพิ่มคำแนะนำการติดตั้ง Dependencies วิธีเรียกใช้สคริปต์ฝึกสอนและประเมินผล เมื่อจัดโครงสร้างโปรเจกต์และสคริปต์เรียบร้อยแล้ว

เอกสารฉบับสมบูรณ์จะระบุการตั้งค่าโมเดลและรายละเอียดการฝึกสอนที่จำเป็น เพื่อให้สามารถทดลองซ้ำได้

---

**Author:** AkanoFah  
**Project:** Deep Learning Capstone: Image Classification
