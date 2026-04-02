# Model Files

Place the trained model weights in this folder before running the app.

Expected files:
- class_Tumor_v2s_clean.keras
- class_Tumor_mobilenet_v3.keras
- class_Tumor_convnext_tiny_tumor.keras
- class_Tumor_densenet_201.keras
- Segmentation_brisc_effunet.keras

The app loads the first three classification models plus the segmentation model by default.
The DenseNet model is kept here for research and training reference.
