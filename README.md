# Linear Regression from Scratch (Notebook)

A from‑scratch linear regression implementation in a single Jupyter notebook using per‑sample gradient descent, with clear loss logging and simple visualization over a CSV dataset.

### Features
- Pure Python notebook with a custom training loop that updates weight and bias per data point.
- Loss printed per epoch and logged per update for quick convergence inspection.
- Scatter vs. fitted line plot for visual validation.
- Minimal dependencies and CPU‑only reproducibility.

### Files
- linear-regression.ipynb — end‑to‑end training, logging, and plotting.
- test_data.csv — required CSV with columns x and y.
- requirements.txt — dependencies for the notebook.

### Installation
- Python 3.12 (or 3.9+) recommended.
- Create and activate a virtual environment, then install:
  - `pip install -r requirements.txt` (pandas, matplotlib, jupyter)
- Launch Jupyter:
  - `jupyter lab` or `jupyter notebook`

### Usage
- Open `linear-regression.ipynb` and run all cells top‑to‑bottom.
- Ensure `test_data.csv` with headers x,y exists in the working directory.
- After training, the notebook prints learned weight and bias and plots the fitted line over the data scatter.

### Math
**Model:**  
`y_hat = b + w * x`

**Per-sample squared loss:**  
`loss = (y - y_hat)^2`

**Gradients:**  
`d(loss)/d(w) = 2*(y_hat - y)*x`  
`d(loss)/d(b) = 2*(y_hat - y)`

**Update rule:**  
`w := w - eta * d(loss)/d(w)`  
`b := b - eta * d(loss)/d(b)`

### Configuration
- Learning rate lr: lower to prevent divergence; increase slightly if convergence is slow.
- Epochs: increase for more passes over the dataset.
- Initialization: defaults w=0, b=0; adjust to experiment with different starting points.

### Data format
- CSV with headers:
  - x: numeric feature.
  - y: numeric target.

### Results and plots
- Prints epoch‑level loss and final parameters (weight and bias).
- Produces:
  - Loss scatter plot: update index vs. squared error.
  - Fit plot: scatter of (x, y) with the learned regression line.

### Extending
- Batch or full‑batch gradient descent for smoother updates.
- Add metrics (MSE, R²) and a train/validation split for evaluation.
- Generalize to multivariate regression with vectorized updates.
- Early stopping by monitoring validation loss.

### Troubleshooting
- Diverging loss: reduce lr or standardize inputs.
- Slow convergence: increase epochs or slightly raise lr; check data scaling/outliers.
- Missing plots: ensure the final plotting cells run and a proper Matplotlib backend is set.

### License
MIT 

### Acknowledgments
Inspired by 3b1b, StatQuest, GregHogg, Google ML Crash course && Muaz.

