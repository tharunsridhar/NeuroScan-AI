# Model Files

This folder is intentionally committed without model weight files.

Download the required `.keras` model files from Hugging Face:

https://huggingface.co/tharunsridhar/brain_tumor_net-ensemble/tree/main/models

After downloading, place the files in this `MODEL/` folder.

Expected files:

- `class_Tumor_v2s_clean.keras`
- `class_Tumor_mobilenet_v3.keras`
- `class_Tumor_convnext_tiny_tumor.keras`
- `Segmentation_brisc_effunet.keras`

The API will report `not_ready` at `/ready` until these files are present locally.
