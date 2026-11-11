# ML_Datasets

> Forked from [ramsha275/ML_Datasets](https://github.com/ramsha275/ML_Datasets). All credit for collecting and sharing these files goes to the original author.

Two small, clean tabular datasets for practicing machine learning. I used them in the deep learning assignments of the PIAIC AI program. See [my_codes_PIAIC](https://github.com/abdulmanankhalid/my_codes_PIAIC).

## Datasets

### 1. `compresive_strength_concrete.csv` — Regression

- **Rows:** 1,030 · **Columns:** 9 · **Missing values:** none
- **Goal:** predict the **compressive strength of concrete (MPa)** from its mixture and age.
- **Features:** amounts (kg per m³ of mixture) of cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate, plus the concrete's age in days.
- **Target:** the last column, concrete compressive strength in MPa.
- **Source:** [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (I-Cheng Yeh).

### 2. `ionosphere_data.csv` — Binary Classification

- **Rows:** 351 · **Columns:** 35 (`feature1` … `feature34`, `label`) · **Missing values:** none
- **Goal:** classify radar returns from the ionosphere as **good (`g`)** or **bad (`b`)**.
- **Class balance:** 225 good / 126 bad.
- **Features:** 34 numeric values from 17 radar pulse readings collected by a system in Goose Bay, Labrador. "Good" returns show evidence of structure in the ionosphere.
- **Source:** [UCI Ionosphere dataset](https://archive.ics.uci.edu/dataset/52/ionosphere) (Johns Hopkins University Applied Physics Laboratory).

## Usage

```python
import pandas as pd

base = "https://raw.githubusercontent.com/abdulmanankhalid/ML_Datasets/main/"

concrete = pd.read_csv(base + "compresive_strength_concrete.csv")
X_c, y_c = concrete.iloc[:, :-1], concrete.iloc[:, -1]

iono = pd.read_csv(base + "ionosphere_data.csv")
X_i = iono.drop(columns="label")
y_i = (iono["label"] == "g").astype(int)   # 1 = good, 0 = bad
```

## Credits

- Original repository: [ramsha275/ML_Datasets](https://github.com/ramsha275/ML_Datasets)
- Data originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
