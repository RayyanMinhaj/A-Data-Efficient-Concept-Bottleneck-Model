# A Data-Efficient Concept Bottleneck Model (DCBM)

Simple project that implements a concept-bottleneck style image classifier using CLIP features.
The pipeline extracts small segment-based visual "concepts", clusters them into a concept bank, trains a linear bottleneck classifier on top of concept activations, and provides a simple explanation for predictions.

---

## Project Structure

```
Cross-Modal-Multi-Task-Learning/
├── src/
|   ├── generate_concepts.py   #segment images, extract CLIP embeddings and build a concept bank (saved to `artifacts/concepts`).    
|   ├── train_cbm.py - #train the concept-bottleneck classifier using the concept bank and ImageNet-like dataset (saves model to `artifacts/models`). 
|   ├── inference.py        #run a demo inference and print an explanation of why a prediction was made.      
|   ├── utils.py       #helper functions (device, loading models, image utils).
├── data/                            
│   ├── imagenette2/                          
├── artifacts/                          
│   ├── models/
│   ├── concepts/      
└──                   
```



## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data: put your dataset in `data/imagenette2` (train/val folders) or update `DATASET_PATH` / constants at the top of the Python scripts.

3. Generate concepts (creates `artifacts/concepts/concept_bank.pt`):

```bash
python 01_generate_concepts.py
```

4. Train the CBM model (saves `artifacts/models/dcbm_model.pth`):

```bash
python 02_train_cbm.py
```

5. Run inference / demo:

```bash
python 03_inference.py
```

---


## What the Output looks like
```
Explaining Image: data/imagenette2/val/n03425413/ILSVRC2012_val_00000732.JPEG
Prediction: n03425413 (88.54%)
------------------------------
Why? Because I saw:
 - Concept 0174 ('gas'): Activation 0.84 (Impact: 1.53)
 - Concept 0417 ('gas'): Activation 0.80 (Impact: 1.27)
 - Concept 0397 ('gas'): Activation 0.82 (Impact: 1.25)
 - Concept 0321 ('instrument'): Activation 0.70 (Impact: 1.11)
 - Concept 0383 ('hose'): Activation 0.74 (Impact: 1.02)
```

Where:
- Prediciton percentage is the confidence of the model's certainty.
- Multiple concepts represent the different centroids (i.e., the different **segments** of an image clustered as one). Sorted by Impact and displayed only the top 5.
- Activation is how clearly the concept appears in the image.
- Impact is (Activation x Weight), telling us how much the concept contributed to the final prediction. 


## Notes
- Scripts use a few hard-coded paths (dataset root, concept bank path, model path). Edit the constants at the top of each script if your files are in different locations.
- `01_generate_concepts.py` and concept creation can be slow — tune `IMGS_PER_CLASS`, `NUM_CLUSTERS`, and `MIN_AREA` at the top of the file for faster iteration.
- The inference demo includes a small vocabulary mapping to name concepts; you can expand it in `03_inference.py`.

## License
Simple permissive use for research/experimentation. No warranty.

---


