# Dataset Setup Guide for EMOTISENSE AI

This project supports standard speech emotion datasets: **RAVDESS** and **TESS**.

## 1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Source**: [Zenodo RAVDESS](https://zenodo.org/record/1188976)
- **Target Directory**: `emotionai/data/ravdess/`
- **Structure**:
  ```
  data/ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
  ```

## 2. TESS (Toronto Emotional Speech Set)
- **Source**: [University of Toronto TSpace](https://tspace.library.utoronto.ca/handle/1807/24487)
- **Target Directory**: `emotionai/data/tess/`
- **Structure**:
  ```
  data/tess/
    ├── OAF_angry/
    │   └── *.wav
    ├── YAF_happy/
    │   └── *.wav
    └── ...
  ```

## Verifying Dataset
Run the preparation script:
```bash
python scripts/prepare_dataset.py --dataset ravdess
python scripts/prepare_dataset.py --dataset tess
```
