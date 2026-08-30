# Dataset Setup & Instructions

This directory holds the **Waste-Weight-10K** dataset files used for training and evaluating the multimodal weight prediction models.

## Expected Directory Layout

Place the dataset CSV and image folders inside `data/` as follows:

```
data/
├── README.md
├── .gitkeep
├── image.csv                   # Main metadata and annotation file
└── <category_folders>/         # Image folders referenced by image.csv
    ├── Appliance/
    ├── Automotive Scrap/
    ├── Battery/
    ├── Cardboard/
    ├── Cylindrical Object/
    ├── Ferrous Metal/
    ├── Rigid Plastic/
    ├── Rubber/
    └── Wood/
```

## Using an External Dataset Path

If you store the dataset on a fast SSD or shared cluster drive outside this repository, you can set the `DATA_PATH` environment variable:

```bash
export DATA_PATH="/path/to/waste_dataset"
python scripts/train.py
```

The system automatically checks:
1. `os.environ["DATA_PATH"]`
2. `./data/`
3. Local development paths

For complete specifications regarding measurements, coordinate systems, and physical setups, see [`docs/DATASET.md`](../docs/DATASET.md).
