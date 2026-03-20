
🌿 CropShield — Real-Time Plant Disease Detection System

## What Problem Does This Solve?

Plant diseases cause up to 40% of global crop losses every year. Farmers in developing countries often lack access to agricultural experts who can diagnose diseases early. By the time a disease is visually obvious, it has already spread.

CropShield solves this by letting any farmer take a photo of a leaf and get an instant diagnosis with treatment advice — no expert required, no internet latency from heavy cloud models.


## The Dataset

PlantVillage Dataset(Kaggle) — 54,305 leaf images across 38 classes covering 14 plant species including Apple, Tomato, Potato, Corn, Grape, Pepper, Peach, and Blueberry. Each class is either a specific disease or a healthy variant.

Why this dataset? :

It is the largest publicly available plant disease image dataset and a well-established benchmark. Results can be compared against published research, which gives my accuracy numbers real meaning.

Train/Val Split: 80% training (43,444 images) / 20% validation (10,861 images) with a fixed random seed (42) for reproducibility.

Data Augmentation applied during training:
- Random horizontal and vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation ±0.3)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Augmentation was applied only to training data — never to validation — to prevent data leakage.


Why EfficientNetB0?

I chose EfficientNetB0 over training from scratch for one reason: transfer learning.

EfficientNetB0 was pretrained on ImageNet (14 million images, 1000 classes). The lower layers already learned to detect edges, textures, and color patterns. Fine-tuning it on plant diseases means the model starts with strong visual feature detectors rather than random weights.

Why not a larger model like EfficientNetB4 or ResNet50?

EfficientNetB0 is the smallest in the EfficientNet family. For a production system where inference speed matters, I wanted the accuracy-to-speed tradeoff optimized. The result: 99.64% accuracy at 21ms inference — a larger model would add latency without meaningful accuracy gain on this dataset.


The Model Pipeline

Input Image (any size)
    ↓
    
Resize to 224×224
    ↓
    
Normalize (ImageNet mean/std)
    ↓
    
EfficientNetB0 backbone (pretrained ImageNet weights)
    ↓
    
Custom classification head (38 output classes)
    ↓
    
Softmax → probability distribution across 38 classes
    ↓
    
Top prediction + confidence score


Training configuration:

- Optimizer: Adam (lr=0.001)
  
- Loss: CrossEntropyLoss
  
- Scheduler: StepLR — halves learning rate every 5 epochs
  
- Epochs: 10
  
- Batch size: 32
  
- Hardware: Google Colab Tesla T4 GPU
  

Training Results

Epoch Train Accuracy Val Accuracy  Note 

 1   94.88%  96.93% Strong start from transfer learning 
 
 2   97.77%  97.15% Fast convergence 
 
 4   98.60%  98.99% Learning rate step kicks in 
 
 6   99.67%  99.64%  Best model saved 
 
 10  99.72%  99.00%  Slight overfitting, epoch 6 used 
 

Best Validation Accuracy: 99.64% at epoch 6

The model was saved at epoch 6, not epoch 10, because validation accuracy peaked there. Using the epoch 10 model would mean deploying a slightly overfit model — a common mistake I deliberately avoided.


Evaluation

Why accuracy is a valid metric here:

The PlantVillage dataset is relatively balanced across classes. Unlike churn prediction (where 74% majority class makes accuracy misleading), here accuracy genuinely reflects model quality.

99.64% means:out of 10,861 validation images, the model misclassified approximately 39 images. Most misclassifications occur between visually similar diseases on the same plant — for example confusing Tomato Early Blight with Tomato Late Blight, which even human experts sometimes struggle with.


Why ONNX Optimization?

PyTorch models carry the full training framework at inference time — unnecessary overhead for production. ONNX (Open Neural Network Exchange) exports just the computation graph.

Result:
Runtime      Avg Inference 

PyTorch        ~340ms 

ONNX Runtime    21ms

16x speed improvement with identical predictions. In a production API serving thousands of requests, this difference is the gap between a usable and unusable system.


Why FastAPI over Flask?

FastAPI is async-native, automatically generates OpenAPI documentation, and has built-in request validation via Pydantic. For an ML inference API that needs to handle concurrent requests, async matters.

Why Docker?

The model runs on Python 3.10 with specific library versions. Without Docker, "it works on my machine" is a real problem. Docker packages the exact environment so the system runs identically on any machine or cloud server.

Challenges and What I Learned

Challenge 1 — ONNX export compatibility

The newer PyTorch ONNX exporter (dynamo-based) generated a split model (`.onnx` + `.onnx.data`). This caused a `file not found` error when loading on a different machine because only the `.onnx` file was copied. Fix: both files must always be kept together.

Challenge 2 — Path encoding on Windows

My username contains a space ("Akhiliny Vijeyagumar"). MLflow encoded it as `%20` in the file path which Windows couldn't resolve. Fix: explicitly set the tracking URI to the full path using `file:///` prefix.

Challenge 3 — Choosing the right epoch

Epoch 10 had higher training accuracy (99.72%) but lower validation accuracy (99.00%) than epoch 6 (99.64%). A naive approach would pick the last epoch. I tracked validation accuracy every epoch and saved the best checkpoint — standard practice but easy to overlook.



Tech Stack


Model : EfficientNetB0 

Framework : PyTorch + timm 

Optimization : ONNX Runtime 

API : FastAPI 

UI : Streamlit 

Container : Docker 

CI/CD : GitHub Actions 

Training : Google Colab T4

Dataset :PlantVillage 

