# Deep Dive

This is some practice questions to sharpen your skills for interview.

## Rules

- Use plain text editor.
- Run with only command line (no debugging tools)

## Structure

```console
├── code (for codes)
├── data (for datasets)
└── README.md
```

## Questions

### Section 1 — Python & NumPy Basics (foundation)

- [ ] **Vector Dot Product** — Cat.: Alg Diff.: E  
  Task: Compute dot product using only elementwise ops + sum (no np.dot).  
  I/O: a (n,), b (n,) → scalar float.  
  Data: Embedded arrays.

- [ ] **Matrix Multiply via Broadcast+Sum** — Cat.: Alg Diff.: E  
  Task: Implement A @ B using broadcasting and sum.  
  I/O: A (m,k), B (k,n) → (m,n).  
  Data: Embedded arrays.

- [ ] **L2 Normalize Vector** — Cat.: Alg Diff.: E  
  Task: Return v / ||v|| with stability for zero vector.  
  I/O: v (n,) → (n,).  
  Data: Embedded arrays.

- [ ] **Min-Max Scale 1D** — Cat.: DS Diff.: E  
  Task: Scale to [0,1] with min/max handling (constant array edge case).  
  I/O: x (n,) → (n,).  
  Data: Embedded arrays.

- [ ] **Euclidean Distance (2D)** — Cat.: Alg Diff.: E  
  Task: Compute distance between two points.  
  I/O: (x1,y1), (x2,y2) → float.  
  Data: Embedded tuples.

- [ ] **Cosine Similarity** — Cat.: Alg Diff.: E  
  Task: Implement cosine sim with epsilon for zeros.  
  I/O: a (n,), b (n,) → float in [-1,1].  
  Data: Embedded.

- [ ] **One-Hot Encode** — Cat.: DS Diff.: E  
  Task: Map integer labels to one-hot matrix.  
  I/O: labels (n,) with K classes → (n,K).  
  Data: Embedded labels.

- [ ] **Softmax (Stable)** — Cat.: ML Diff.: E  
  Task: Implement numerically stable softmax along last axis.  
  I/O: logits (...,K) → probs shape (...,K).  
  Data: Embedded.

- [ ] **Cross-Entropy Loss (Vectorized)** — Cat.: ML Diff.: M  
  Task: CE for one-hot or index labels; average over batch.  
  I/O: probs (N,K), y (N,) or (N,K) → scalar.  
  Data: Embedded.

- [ ] **Train/Test Split (No sklearn)** — Cat.: DS Diff.: E  
  Task: Shuffle + split with seed and stratify option.  
  I/O: (X,y) → (X_tr,y_tr,X_te,y_te).  
  Data: Embedded.

- [ ] **Z-Score Normalization** — Cat.: DS Diff.: E  
  Task: Standardize features; keep mean/std for inverse transform.  
  I/O: X (N,D) → (N,D), plus (mean,std).  
  Data: Embedded.

- [ ] **Argmax Tie-Break** — Cat.: Alg Diff.: E  
  Task: Argmax returning smallest index on ties.  
  I/O: v (n,) → int index.  
  Data: Embedded.

- [ ] **Top-k (Partial Selection)** — Cat.: Alg Diff.: M  
  Task: Return top-k values + indices without full sort (np.argpartition).  
  I/O: v (n,), k → (values (k,), idx (k,)).  
  Data: Embedded.

- [ ] **Im2col for 2D Convolution** — Cat.: DL Diff.: M  
  Task: Implement im2col(X, kernel, stride, pad).  
  I/O: X (N,C,H,W) → (N, C*kh*kw, L)  
  Data: Embedded.

  - [ ] **Col2im (Inverse)** — Cat.: DL Diff.: M  
    Task: Inverse of Q14 with overlap-add.  
    I/O: columns → reconstructed X.  
    Data: Use same as Q14.

---

### Section 2 — Classical ML (from scratch)

- [ ] **Linear Regression (Normal Eq.)** — Cat.: ML Diff.: M  
  Task: Closed-form w = (X^T X)^(-1) X^T y with regularization option.  
  I/O: X (N,D), y (N,) → w (D,).  
  Data: Synthetic linear.

- [ ] **Linear Regression (GD)** — Cat.: ML Diff.: M  
  Task: Batch gradient descent with learning rate & epochs.  
  I/O: same → w.  
  Data: same as Q16.

- [ ] **Polynomial Basis Expansion** — Cat.: ML Diff.: M  
  Task: Map 1D x to [x, x^2, …, x^p]; fit ridge regression.  
  I/O: x (N,), degree p → w (p,).  
  Data: Synthetic cubic.

- [ ] **Logistic Regression (Binary)** — Cat.: ML Diff.: M  
  Task: Sigmoid + CE loss + GD/SGD; predict prob & class.  
  I/O: X (N,D), y∈{0,1} → w (D,).  
  Data: Synthetic separable.

- [ ] **Regularized Logistic (L2)** — Cat.: ML Diff.: M  
  Task: Add L2 penalty; compare to Q19 by accuracy/AUC.  
  I/O: same.  
  Data: same.

- [ ] **Naive Bayes (Gaussian)** — Cat.: ML Diff.: M  
  Task: Fit class means/vars/prior; predict via log-likelihoods.  
  I/O: X (N,D), y (N,) → preds.  
  Data: Iris CSV (sepal_len, sepal_wid, petal_len, petal_wid, class).

- [ ] **K-NN Classifier** — Cat.: ML Diff.: M  
  Task: L2 distance, majority vote with tie rule; try k∈{1,3,5}.  
  I/O: X_tr, y_tr, X_te → y_pred.  
  Data: Iris CSV.

- [ ] **Decision Tree (ID3/Gini)** — Cat.: ML Diff.: H  
  Task: CART-style binary splits; max depth & min samples.  
  I/O: X (N,D), y (N,) → tree structure + predict.  
  Data: Titanic CSV (pclass, sex, age, sibsp, parch, fare, embarked, survived).

- [ ] **Random Forest (Bagging)** — Cat.: ML Diff.: H  
  Task: Bagged trees with feature subsampling; OOB score.  
  I/O: Train set → ensemble predictions.  
  Data: Titanic CSV.

- [ ] **SVM (Linear, Hinge Loss, SGD)** — Cat.: ML Diff.: H  
  Task: Implement primal SGD for hinge loss with L2.  
  I/O: X,y∈{-1,1} → w,b.  
  Data: Synthetic 2D.

- [ ] **PCA via SVD** — Cat.: ML Diff.: M  
  Task: Center, SVD, project to k comps; explained variance.  
  I/O: X (N,D), k → Z (N,k).  
  Data: Wine CSV (UCI Wine, features + target).

- [ ] **t-SNE (Simplified)** — Cat.: ML Diff.: H  
  Task: Perplexity → P matrix, gradient descent on KL; 2D embedding.  
  I/O: X (N,D) → Y (N,2).  
  Data: MNIST sample (10k rows, images flattened).

- [ ] **K-Means** — Cat.: ML Diff.: M  
  Task: Lloyd’s algorithm, k-means++ init; inertia curve.  
  I/O: X (N,D), k → centroids (k,D), labels (N,).  
  Data: Iris or synthetic blobs.

- [ ] **DBSCAN** — Cat.: ML Diff.: H  
  Task: Core/reachable/noise; BFS/union-find; eps/minPts.  
  I/O: X (N,D) → cluster labels (-1 for noise).  
  Data: 2D moons (synthetic).

  - [ ] **Gaussian Mixture Model (EM)** — Cat.: ML Diff.: H  
    Task: E/M steps; log-likelihood tracking; k components.  
    I/O: X (N,D) → π, μ, Σ.  
    Data: Synthetic Gaussians.

---

### Section 3 — Data Science / Analysis

- [ ] **Data Cleaning Pipeline** — Cat.: DS Diff.: E  
  Task: Drop dupes/NA, fix dtypes, basic sanity checks.  
  I/O: CSV → clean DataFrame.  
  Data: Small sales CSV (id,date,region,amount).

- [ ] **GroupBy Aggregations** — Cat.: DS Diff.: E  
  Task: Sum/mean/count by region and by region,month.  
  I/O: DF → summary DF.  
  Data: Same as Q31.

- [ ] **Pivot Table** — Cat.: DS Diff.: M  
  Task: Pivot sales by product (rows) × month (cols), mean amount.  
  I/O: DF → pivot DF.  
  Data: Sales CSV (date,product,amount).

- [ ] **Joins (Merge)** — Cat.: DS Diff.: E  
  Task: Inner/left/right joins on customer_id.  
  I/O: two CSVs → merged DF.  
  Data: customers.csv, orders.csv.

- [ ] **Correlation Matrix + Heatmap (values only)** — Cat.: DS Diff.: E  
  Task: Compute corr and print nicely (no plotting libs required).  
  I/O: DF → (D,D) matrix.  
  Data: Wine CSV.

- [ ] **Time Series Resample** — Cat.: DS Diff.: M  
  Task: Convert daily to monthly sum; handle missing days.  
  I/O: DF → resampled DF.  
  Data: Web traffic CSV (date, visits).

- [ ] **Rolling Stats** — Cat.: DS Diff.: M  
  Task: Compute 7-day moving avg/std; edge handling.  
  I/O: series → series.  
  Data: same as Q36.

- [ ] **Outlier Detection (Z-Score / IQR)** — Cat.: DS Diff.: M  
  Task: Flag & optionally clip/fill outliers.  
  I/O: series → mask/cleaned series.  
  Data: same.

- [ ] **Missing Data Imputation** — Cat.: DS Diff.: M  
  Task: Mean/median/ffill/bfill; evaluate on synthetic masks.  
  I/O: series/DF → filled data.  
  Data: any numeric DF.

- [ ] **Feature Hashing** — Cat.: DS Diff.: M  
  Task: Hash trick for large categorical features.  
  I/O: list of strings → sparse/dense hashed features.  
  Data: embedded strings.

- [ ] **Target Encoding (K-fold)** — Cat.: DS Diff.: H  
  Task: Leakage-safe K-fold mean target enc.  
  I/O: X_cat, y → encoded feature.  
  Data: Adult CSV (UCI Adult: age, workclass, education, ... , income).

- [ ] **Text Tokenize + TF-IDF (from scratch)** — Cat.: DS Diff.: H  
  Task: Build vocab, counts, IDF, TF-IDF matrix.  
  I/O: list of docs → (N,V) matrix.  
  Data: 20 Newsgroups (plain text, filenames & content).

- [ ] **A/B Test (z-test / t-test)** — Cat.: DS Diff.: M  
  Task: Implement two-sample tests; compute p-value, CI.  
  I/O: groupA, groupB arrays → stats.  
  Data: Embedded arrays.

- [ ] **ROC & AUC (from scratch)** — Cat.: DS Diff.: H  
  Task: Compute ROC points and trapezoid AUC.  
  I/O: scores (N,), labels (N,) → AUC float.  
  Data: Embedded.

  - [ ] **Calibration Curve + Brier Score** — Cat.: DS Diff.: H  
    Task: Bin probs, reliab. diagram data, Brier metric.  
    I/O: probs, labels → curve data + score.  
    Data: Embedded.

---

### Section 4 — Intermediate ML / Feature Engineering

- [ ] **Regularization Paths (Ridge/Lasso)** — Cat.: ML Diff.: H  
  Task: Coordinate descent for Lasso; closed-form ridge; plot path values (print table).  
  I/O: X,y → w per λ.  
  Data: Boston Housing CSV (features + medv).

- [ ] **Gradient Checking** — Cat.: ML Diff.: M  
  Task: Finite-diff check for any loss function; report max abs diff.  
  I/O: f(w), grad(w) → diff scalar.  
  Data: Embedded.

- [ ] **Bias-Variance Decomposition (Sim)** — Cat.: ML Diff.: H  
  Task: Simulate fits over many noise draws; estimate bias/var.  
  I/O: returns scalars.  
  Data: Synthetic polynomial.

- [ ] **Feature Scaling Impact** — Cat.: DS Diff.: M  
  Task: Compare GD convergence with raw vs standardized.  
  I/O: logs & final error.  
  Data: Synthetic.

  - [ ] **Learning Curves** — Cat.: ML Diff.: M  
    Task: Train sizes vs error for model; print table.  
    I/O: size → train/val loss.  
    Data: Iris or Wine.

---

### Section 5 — PyTorch Basics

- [ ] **Single Neuron (No Autograd)** — Cat.: DL Diff.: M  
  Task: Manual forward + manual gradient update on MSE.  
  I/O: X (N,D), y (N,) → trained w,b.  
  Data: Synthetic linear.

- [ ] **Two-Layer MLP (Autograd)** — Cat.: DL Diff.: M  
  Task: ReLU hidden, CE loss; training loop with torch.no_grad() where needed.  
  I/O: X (N,D), y (N,) → accuracy.  
  Data: Iris.

- [ ] **Custom Loss (Focal Loss)** — Cat.: DL Diff.: H  
  Task: Implement focal loss module; compare to CE on class-imbalance.  
  I/O: logits, targets → scalar loss.  
  Data: Imbalanced synthetic.

- [ ] **Weight Initialization Study** — Cat.: DL Diff.: M  
  Task: Compare Xavier/He/random on MLP; report train/val curves (print).  
  I/O: logs.  
  Data: MNIST (subset).

  - [ ] **Manual Optimizer (SGD+Momentum)** — Cat.: DL Diff.: M  
    Task: Implement optimizer step manually; plug into MLP training.  
    I/O: parameters → updated parameters.  
    Data: From Q52.

---

### Section 6 — Deep Learning Architectures

- [ ] **Conv2D from Scratch (using im2col)** — Cat.: DL Diff.: H  
  Task: Forward conv via Q14; check vs PyTorch conv2d.  
  I/O: X (N,C,H,W), W (F,C,kh,kw) → (N,F,Ho,Wo).  
  Data: Synthetic images.

- [ ] **LeNet-5** — Cat.: DL Diff.: M  
  Task: Implement LeNet and train on MNIST; report test acc.  
  I/O: train loop → accuracy %.  
  Data: MNIST (28×28 grayscale; split train/test).

- [ ] **VGG-16 (Configurable Blocks)** — Cat.: DL Diff.: H  
  Task: Build VGG blocks; train on CIFAR-10 (small epochs).  
  I/O: final acc %.  
  Data: CIFAR-10 (32×32 color; train/test).

- [ ] **ResNet-18 (BasicBlock)** — Cat.: DL Diff.: H  
  Task: Implement residual blocks + downsampling; train on CIFAR-10.  
  I/O: acc %.  
  Data: CIFAR-10.

- [ ] **BatchNorm vs LayerNorm** — Cat.: DL Diff.: H  
  Task: Add BN and LN to same MLP; compare convergence.  
  I/O: logs + final acc.  
  Data: MNIST.

- [ ] **RNN (vanilla) Char-Level** — Cat.: DL Diff.: H  
  Task: Implement tanh RNN cell; next-char prediction.  
  I/O: text → predicted sequence.  
  Data: Tiny Shakespeare (plain text).

- [ ] **LSTM from Scratch (Cell math)** — Cat.: DL Diff.: A  
  Task: Implement LSTM cell equations with PyTorch ops; train on next-word.  
  I/O: sequence → next token.  
  Data: Penn Treebank small (tokenized text).

- [ ] **GRU from Scratch** — Cat.: DL Diff.: A  
  Task: Implement GRU cell; compare to LSTM on same data.  
  I/O: perplexity.  
  Data: same as Q62.

- [ ] **Transformer Encoder Block** — Cat.: DL Diff.: A  
  Task: Multi-head self-attn, residual, LN, FFN; mask support.  
  I/O: X (N,T,D) → same shape.  
  Data: Synthetic.

- [ ] **Masked LM (Mini-BERT)** — Cat.: DL Diff.: A  
  Task: Train small transformer on masked tokens.  
  I/O: tokenized text → MLM loss.  
  Data: WikiText-2 (plain text).

- [ ] **GAN (MLP) on MNIST** — Cat.: DL Diff.: H  
  Task: Basic GAN loop; show generated sample grid (save PNG).  
  I/O: noise → images.  
  Data: MNIST.

- [ ] **DCGAN (Conv) on CIFAR-10** — Cat.: DL Diff.: A  
  Task: Conv generator/discriminator; save samples.  
  I/O: noise → 32×32 images.  
  Data: CIFAR-10.

- [ ] **U-Net (Segmentation)** — Cat.: DL Diff.: A  
  Task: Build U-Net; train on small Carvana or Oxford-IIIT Pet segmentation subset (images + masks, PNG).  
  I/O: image → mask; Dice/IoU.

- [ ] **Attention as Matrix Ops (from scratch)** — Cat.: DL Diff.: H  
  Task: Implement scaled dot-product attention (Q,K,V) + masking.  
  I/O: Q,K,V → attended outputs.  
  Data: Embedded tensors.

  - [ ] **Seq2Seq w/ Attention (Toy Translation)** — Cat.: DL Diff.: A  
    Task: Char-level encoder-decoder attention model.  
    I/O: toy pairs → translated output.  
    Data: Small parallel toy set (e.g., digit ↔ word).

---

### Section 7 — Advanced Algorithms (interview-oriented)

- [ ] **Dijkstra (Binary Heap)** — Cat.: Alg Diff.: M  
  Task: Shortest path on weighted graph.  
  I/O: adjacency list → distances.  
  Data: Embedded graph.

- [ ] **A* Pathfinding (Grid)** — Cat.: Alg Diff.: M  
  Task: Manhattan heuristic; return path.  
  I/O: grid with obstacles → path coords.  
  Data: Embedded grid.

- [ ] **Topological Sort + Cycle Detection** — Cat.: Alg Diff.: M  
  Task: Kahn’s algorithm; detect cycles.  
  I/O: DAG edges → topo order.  
  Data: Embedded edges.

- [ ] **Union-Find (DSU)** — Cat.: Alg Diff.: E  
  Task: Path compression + union by rank.  
  I/O: ops stream → resulting sets.  
  Data: Embedded.

- [ ] **PageRank (Power Iteration)** — Cat.: Alg/ML Diff.: H  
  Task: Damping, dangling handling, tolerance stop.  
  I/O: sparse graph → ranks.  
  Data: Small web graph CSV (src,dst).

- [ ] **Apriori (Association Rules)** — Cat.: DS Diff.: H  
  Task: Frequent itemsets + rules with support/conf/lift.  
  I/O: transactions list → rules table.  
  Data: Groceries CSV (transaction_id,item).

- [ ] **HMM Forward-Backward** — Cat.: ML Diff.: H  
  Task: Prob inference with log-space stability.  
  I/O: params + obs → α/β + likelihood.  
  Data: Synthetic HMM.

- [ ] **Viterbi Decoding** — Cat.: ML Diff.: H  
  Task: Most likely hidden state sequence.  
  I/O: params + obs → state path.  
  Data: same as Q77.

- [ ] **Simulated Annealing (TSP)** — Cat.: Alg Diff.: A  
  Task: SA schedule; 2-opt moves; distance reduction.  
  I/O: city coords → tour length.  
  Data: Synthetic city list.

  - [ ] **Matrix Factorization (ALS)** — Cat.: ML Diff.: A  
    Task: Implicit/explicit ALS; RMSE evaluation.  
    I/O: rating triples (u,i,r) → U,V.  
    Data: MovieLens 100k CSV (userId,movieId,rating,timestamp).

---

### Section 8 — Integrated Projects (compose skills)

- [ ] **MNIST Classifier (CNN baseline)** — Cat.: DL Diff.: M  
  Task: 2-3 conv layers + MLP; >98% test acc target.  
  I/O: model → accuracy.  
  Data: MNIST.

- [ ] **IMDB Sentiment (LSTM/GRU)** — Cat.: DL Diff.: H  
  Task: Tokenize, pad, embed, LSTM; accuracy/F1.  
  I/O: texts → labels.  
  Data: IMDB (train/test, text + label).

- [ ] **Stock Forecast (LSTM)** — Cat.: DL Diff.: H  
  Task: Sliding window; predict next close; MAE/MAPE.  
  I/O: prices CSV → predictions.  
  Data: Any daily stock CSV (date,open,high,low,close,volume).

- [ ] **Face Embeddings + KNN** — Cat.: DL Diff.: A  
  Task: Train small CNN to produce embeddings; classify with KNN.  
  I/O: image → identity label.  
  Data: LFW (images, identities).

- [ ] **Speech Commands (MFCC+CNN)** — Cat.: DL Diff.: A  
  Task: Extract MFCC (numpy) + CNN; accuracy.  
  I/O: WAV → label.  
  Data: Google Speech Commands (WAV, label).

- [ ] **Tabular AutoML-lite** — Cat.: ML/DS Diff.: A  
  Task: Try LR, RF, XGBoost-like (use RF proxy), simple CV, pick best.  
  I/O: dataset path → best model & score.  
  Data: Adult CSV.

- [ ] **Anomaly Detection (Autoencoder)** — Cat.: DL Diff.: H  
  Task: Dense AE on tabular; threshold by recon error.  
  I/O: X → anomaly mask.  
  Data: KDD Cup 99 small or synthetic.

- [ ] **Image Augmentations (from scratch)** — Cat.: DL Diff.: M  
  Task: Flip, crop, color jitter (numpy/torch ops).  
  I/O: image → augmented image.  
  Data: CIFAR-10 subset.

- [ ] **Hyperparam Search (Grid/Random)** — Cat.: ML Diff.: M  
  Task: Implement simple search with seed reproducibility.  
  I/O: search space → best params/score.  
  Data: any from above.

  - [ ] **Model Checkpointing & Resume** — Cat.: DL Diff.: M  
    Task: Save/load weights & optimizer state; resume training.  
    I/O: path → restored trainer.  
    Data: use Q81/82 models.

---

### Section 9 — Multi-Step Builders (depend on prior pieces)

- [ ] **End-to-End Tabular Pipeline** — Cat.: DS/ML Diff.: H  
  Task: Build: clean → encode cat (Q45) → scale (Q11) → model (Q19/24) → CV → test.  
  I/O: CSV path → test score + artifacts.  
  Data: Adult CSV (train/test split: 80/20).

- [ ] **Text Pipeline: Token→TF-IDF→LogReg** — Cat.: DS/ML Diff.: H  
  Task: Use Q42 TF-IDF + Q20 logistic; evaluate F1.  
  I/O: raw texts → predictions.  
  Data: 20 Newsgroups.

- [ ] **Image Pipeline: Augment→CNN→Calibrate** — Cat.: DL Diff.: A  
  Task: Q88 aug → Q81 CNN → Q45 calibration; report ECE/Brier.  
  I/O: model + metrics.  
  Data: CIFAR-10.

- [ ] **Sequence LM: RNN→LSTM→Transformer Comparison** — Cat.: DL Diff.: A  
  Task: Train small models (Q61, Q62, Q64/65) on same text; compare ppl/perf vs params.  
  I/O: table summary.  
  Data: WikiText-2 small.

- [ ] **Recsys: ALS→KNN Hybrid** — Cat.: ML Diff.: A  
  Task: Combine ALS (Q80) latent preds with item-KNN fallback; RMSE/Recall@K.  
  I/O: ratings → metrics.  
  Data: MovieLens 100k.

- [ ] **Clustering + Semi-Supervised** — Cat.: ML Diff.: A  
  Task: K-Means (Q28) to label high-confidence, train classifier on partial labels, self-train.  
  I/O: semi-sup accuracy.  
  Data: CIFAR-10 (use subset labels).

- [ ] **Time Series: Prophet-like Features (manual)** — Cat.: DS/ML Diff.: H  
  Task: Build Fourier seasonality + holiday regressors + LR; compare to LSTM (Q83).  
  I/O: forecasts + MAE.  
  Data: daily web traffic CSV.

- [ ] **Production-Style Metrics & Drift** — Cat.: DS Diff.: H  
  Task: Compute PSI/KS drift between train vs recent batch; trigger alert.  
  I/O: two DataFrames → drift report.  
  Data: Synthetic splits.

- [ ] **Explainability: Grad-CAM (from scratch)** — Cat.: DL Diff.: A  
  Task: Implement Grad-CAM for CNN; save heatmaps.  
  I/O: image → heatmap PNG.  
  Data: CIFAR-10 sample images.

  - [ ] **Tiny MLOps Loop (offline)** — Cat.: DS/ML Diff.: A  
    Task: Scripted pipeline: load data → preprocess → train → evaluate → save model + metrics JSON + versioned config; re-run with different config and compare.  
    I/O: artifacts saved to disk (.pt or .npz, .json).  
    Data: pick any prior dataset (recommend Adult or MNIST).

---

### Section 10 — Final 30: Interview-Style, Advanced & Composite

- [ ] **Regularized Logistic with Class Weights** — Cat.: ML Diff.: M  
  Task: Add class weights to CE; tune λ via CV.  
  I/O: X,y → best w.  
  Data: Imbalanced binary CSV (create from Adult).

- [ ] **Multiclass Softmax Regression** — Cat.: ML Diff.: M  
  Task: Softmax + CE for K classes; report macro-F1.  
  I/O: X,y∈{0..K-1} → W (D,K).  
  Data: Iris (3 classes) or MNIST.

- [ ] **One-Vs-Rest Linear SVM** — Cat.: ML Diff.: H  
  Task: Train K linear SVMs with hinge loss SGD; predict argmax.  
  I/O: X,y → multiclass preds.  
  Data: MNIST (flattened).

- [ ] **Calibration: Platt Scaling (from scratch)** — Cat.: DS Diff.: H  
  Task: Fit sigmoid on val set; apply to test; compare Brier.  
  I/O: scores → calibrated probs.  
  Data: any classifier scores from Q81/93.

- [ ] **Learning Rate Schedules** — Cat.: DL Diff.: M  
  Task: Step, cosine, warmup; track effect on training loss.  
  I/O: logs table.  
  Data: MNIST.

- [ ] **Label Smoothing (CE)** — Cat.: DL Diff.: H  
  Task: Add label smoothing to CE; compare to vanilla on overfit.  
  I/O: accuracy & calibration.  
  Data: CIFAR-10 subset.

- [ ] **Mixup/CutMix Implementations** — Cat.: DL Diff.: A  
  Task: Implement both; evaluate robustness to label noise.  
  I/O: acc under noisy labels.  
  Data: CIFAR-10.

- [ ] **Weight Decay vs L2 Penalty (AdamW vs Adam)** — Cat.: DL Diff.: H  
  Task: Compare true decoupled weight decay vs L2; report gaps.  
  I/O: logs + final acc.  
  Data: MNIST/CIFAR-10.

- [ ] **Cosine Similarity Search (ANN Baseline)** — Cat.: Alg Diff.: H  
  Task: Brute-force nearest neighbors for embeddings; optimize with batching.  
  I/O: query matrix → top-k ids.  
  Data: Use embeddings from Q84.

- [ ] **Graph Convolution (GCN layer)** — Cat.: DL Diff.: A  
  Task: Implement Kipf-Welling GCN layer (normalized adjacency).  
  I/O: X (N,F), A (N,N) → (N,F').  
  Data: Cora citation network (edge list, features, labels).

- [ ] **Node Classification with 2-Layer GCN** — Cat.: DL Diff.: A  
  Task: Train GCN from Q110; report accuracy.  
  I/O: logits → preds.  
  Data: Cora.

- [ ] **Triplet Loss for Image Embeddings** — Cat.: DL Diff.: A  
  Task: Implement triplet loss with semi-hard mining.  
  I/O: images → embedding vectors.  
  Data: CIFAR-10 (treat class as identity).

- [ ] **Contrastive Learning (SimCLR-lite)** — Cat.: DL Diff.: A  
  Task: Two augmented views, projection head, NT-Xent loss.  
  I/O: embeddings → linear eval acc.  
  Data: CIFAR-10.

- [ ] **Knowledge Distillation (Teacher→Student)** — Cat.: DL Diff.: A  
  Task: Train small student on teacher soft targets (temperature).  
  I/O: student acc vs teacher.  
  Data: Teacher from Q59; student shallow CNN.

- [ ] **Quantization-Aware Training (8-bit emu)** — Cat.: DL Diff.: A  
  Task: Fake quantize weights/activations; accuracy drop analysis.  
  I/O: quantized model acc.  
  Data: MNIST/CIFAR-10.

- [ ] **Pruning (Magnitude)** — Cat.: DL Diff.: H  
  Task: Global unstructured pruning; sparsity vs accuracy curve.  
  I/O: table of sparsity→acc.  
  Data: MNIST model.

- [ ] **Saliency Maps (∂y/∂x)** — Cat.: DL Diff.: H  
  Task: Compute vanilla gradients wrt input; visualize as array.  
  I/O: image → saliency array (save PNG optional).  
  Data: CIFAR-10.

- [ ] **SHAP-like (KernelSHAP approx)** — Cat.: DS Diff.: A  
  Task: Approximate Shapley values via weighted linear explainer.  
  I/O: instance → feature attributions.  
  Data: Adult.

- [ ] **Feature Selection (Mutual Information)** — Cat.: DS/ML Diff.: H  
  Task: Histogram-based MI estimate; pick top features.  
  I/O: feature indices.  
  Data: KDD small or Adult.

- [ ] **Time-Aware Train/Val Split** — Cat.: DS Diff.: M  
  Task: Rolling origin evaluation for time series.  
  I/O: splits dict.  
  Data: daily traffic CSV.

- [ ] **ARIMA-lite (Yule-Walker)** — Cat.: ML Diff.: A  
  Task: Estimate AR(p) via Yule-Walker; forecast h steps.  
  I/O: series → forecast.  
  Data: time series CSV.

- [ ] **Reinforcement Learning: Bandits (ε-greedy, UCB)** — Cat.: ML Diff.: H  
  Task: Simulate multi-armed bandit; cumulative regret.  
  I/O: logs & regret plot data.  
  Data: synthetic arm rewards.

- [ ] **Policy Gradient (REINFORCE) on GridWorld** — Cat.: DL Diff.: A  
  Task: Stochastic policy network; baseline; returns increase.  
  I/O: avg return vs episodes.  
  Data: synthetic GridWorld.

- [ ] **Beam Search Decoder** — Cat.: Alg/DL Diff.: H  
  Task: Implement beam search for seq models; compare to greedy.  
  I/O: step function → decoded sequence.  
  Data: use Q70 model.

- [ ] **CTC Loss (forward-backward)** — Cat.: DL Diff.: A  
  Task: Implement CTC DP and train tiny model on toy speech text alignments.  
  I/O: loss scalar.  
  Data: synthetic sequences.

- [ ] **Neural ODE (Euler/ RK4 solver)** — Cat.: DL Diff.: A  
  Task: Implement ODEBlock with numeric solver; fit toy 2D spirals.  
  I/O: embeddings.  
  Data: synthetic spirals.

- [ ] **Bayesian Linear Regression (Conjugate)** — Cat.: ML Diff.: H  
  Task: Compute posterior mean/cov; predictive distribution.  
  I/O: posterior params + predictive mean/var.  
  Data: synthetic.

- [ ] **Variational Autoencoder (VAE)** — Cat.: DL Diff.: A  
  Task: Reparam trick; train on MNIST; sample images.  
  I/O: recon loss + KL, samples.

- [ ] **Sparse Coding (L1) with ISTA** — Cat.: ML Diff.: A  
  Task: Learn dictionary and sparse codes alternatingly.  
  I/O: dictionary, codes.  
  Data: small natural image patches.

- [ ] **Optimal Transport (Sinkhorn-Knopp)** — Cat.: ML Diff.: A  
  Task: Entropic OT between two histograms; cost matrix.  
  I/O: transport plan & cost.  
  Data: synthetic distributions.

- [ ] **Meta-Learning: MAML-lite** — Cat.: DL Diff.: A  
  Task: 2-step inner loop on few-shot classification; outer update.  
  I/O: few-shot accuracy.  
  Data: Omniglot subset.

- [ ] **Curriculum Learning Schedule** — Cat.: DL Diff.: H  
  Task: Start with easy samples → gradually harder; compare training dynamics.  
  I/O: loss curves diff.  
  Data: CIFAR-10 with custom difficulty heuristic.

- [ ] **Data Loader from Scratch** — Cat.: DS/DL Diff.: M  
  Task: Iterable that yields mini-batches with shuffling & workerless prefetch.  
  I/O: yields (X,y).  
  Data: any CSV.

- [ ] **FP16 Mixed Precision (manual scaler)** — Cat.: DL Diff.: A  
  Task: Cast to half, maintain master FP32 weights, gradient scaling.  
  I/O: stable training logs.  
  Data: MNIST/CIFAR-10.

- [ ] **Causal Inference: Propensity Score Matching** — Cat.: DS Diff.: A  
  Task: Estimate propensity (logistic), match treated/control, ATE.  
  I/O: ATE estimate + balance checks.  
  Data: Lalonde dataset CSV.

- [ ] **Counterfactual Fairness Metric** — Cat.: DS/ML Diff.: A  
  Task: Compute demographic parity & equalized odds; simple reweighting.  
  I/O: metrics pre/post.  
  Data: Adult.

- [ ] **Active Learning Loop (Uncertainty Sampling)** — Cat.: ML Diff.: A  
  Task: Start with small labeled set; iteratively query most uncertain.  
  I/O: accuracy vs labeled size.  
  Data: CIFAR-10 subset.

- [ ] **Currying Earlier Models: Use Pretrained from Q59 for Few-Shot** — Cat.: DL Diff.: A  
  Task: Freeze backbone; train linear head on k-shot per class.  
  I/O: few-shot acc.  
  Data: CIFAR-10.

- [ ] **Ensembling (Bagging + Averaging + Logit-Avg)** — Cat.: ML/DL Diff.: H  
  Task: Train multiple seeds/models; combine predictions; compare.  
  I/O: acc improvements.  
  Data: CIFAR-10.

  - [ ] **Robustness: FGSM Attack + Adversarial Training** — Cat.: DL Diff.: A  
    Task: Implement FGSM; measure clean/adv acc; adversarially train small model.  
    I/O: accuracies pre/post.  
    Data: MNIST/CIFAR-10.

---

## Dataset Notes & Formats (for quick reference)

- **Iris** — CSV with columns: `sepal_length,sepal_width,petal_length,petal_width,species`.  
- **Wine** — CSV numeric features + target (class).  
- **Titanic** — CSV cols: `pclass,sex,age,sibsp,parch,fare,embarked,survived`.  
- **Adult (UCI)** — CSV cols include `age,workclass,education,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income`.  
- **MNIST** — images (28×28) + labels; can be loaded or provided as PNG folders.  
- **CIFAR-10** — 32×32 RGB images + labels (10 classes).  
- **20 Newsgroups** — plain text files; label = directory name.  
- **IMDB** — train/test text + binary labels.  
- **MovieLens 100k** — `userId,movieId,rating,timestamp`.  
- **WikiText-2 / Penn Treebank / Tiny Shakespeare** — plain text.  
- **Cora** — citation network: cites edge list, node features, labels.  
- **Google Speech Commands** — WAV audio files + label folders.  
- **Oxford-IIIT Pet / Carvana** — images + binary/instance masks (PNG).  
- **Lalonde** — treatment effect study table (`treat`, outcomes, covariates).
