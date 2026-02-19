# Chapter 206: Self-Distillation Trading

## 1. Introduction

Self-distillation is a remarkable paradigm in which a model teaches itself by using its own predictions as soft targets. Unlike traditional knowledge distillation, where a large teacher model transfers knowledge to a smaller student, self-distillation eliminates the need for a separate teacher entirely. The model serves as both teacher and student, iteratively refining its own representations through successive generations of training.

The intuition behind self-distillation is deceptively simple: when a neural network produces probability distributions over classes (soft targets), those distributions contain richer information than hard labels alone. A prediction of [0.7, 0.2, 0.1] for a three-class problem tells us not only that the first class is most likely, but also encodes relationships between classes. By training a new copy of the model on these soft targets, we transfer this "dark knowledge" — the implicit structure captured in the probability distribution — back into the architecture.

In trading, self-distillation addresses a fundamental challenge: we often have limited labeled data, and the labels themselves are noisy. Market regime labels (bull, bear, sideways) are inherently fuzzy — a market can be 70% bullish and 30% sideways. Hard labels destroy this nuance, but soft targets from a trained model can preserve it. Self-distillation allows us to iteratively improve a trading model without requiring additional data, external teachers, or ensemble methods, making it particularly valuable for production systems where simplicity and latency matter.

This chapter explores self-distillation from its mathematical foundations through practical implementation in Rust, with real market data from the Bybit exchange.

## 2. Mathematical Foundation

### Born-Again Networks

The concept of Born-Again Networks (BAN), introduced by Furlanello et al. (2018), provides the theoretical backbone for self-distillation. The process works as follows:

1. Train an initial model M_0 on hard labels using standard cross-entropy loss
2. Use M_0 to generate soft predictions on the training data
3. Train a new model M_1 (identical architecture) using a combination of hard labels and soft targets from M_0
4. Repeat: use M_k to train M_{k+1}

The loss function for generation k+1 combines two terms:

```
L_total = alpha * L_CE(y, p_{k+1}) + (1 - alpha) * L_KD(p_k, p_{k+1})
```

where:
- L_CE(y, p_{k+1}) is the standard cross-entropy loss between hard labels y and the new model's predictions p_{k+1}
- L_KD(p_k, p_{k+1}) is the knowledge distillation loss between the previous generation's soft targets and the new model's predictions
- alpha controls the balance between hard and soft supervision

### Self-Training Iterations

Each generation of self-distillation can be viewed as a form of regularized re-training. The soft targets from generation k act as a smoothed version of the training labels, providing several benefits:

- **Implicit label smoothing**: Soft targets naturally prevent overconfident predictions
- **Inter-class relationship preservation**: The model learns which classes are "close" to each other
- **Noise reduction**: Averaging over model uncertainty helps filter out label noise

Formally, if we denote the soft predictions of generation k as:

```
q_k(c | x) = exp(z_k(c | x) / T) / sum_j exp(z_k(j | x) / T)
```

where z_k are the logits and T is the temperature parameter, then the KL divergence loss becomes:

```
L_KD = T^2 * sum_c q_k(c | x) * log(q_k(c | x) / q_{k+1}(c | x))
```

The T^2 scaling factor compensates for the gradient magnitude reduction caused by the temperature scaling.

### Progressive Self-Distillation

Progressive self-distillation extends the basic framework by gradually increasing the difficulty of the training signal across generations. In early generations, a high temperature (T = 5-10) produces very soft targets that emphasize inter-class relationships. In later generations, the temperature is lowered (T = 1-2), pushing the model toward sharper, more confident predictions.

This curriculum-like approach helps the model first learn the broad structure of the problem before focusing on decision boundaries:

```
T_k = T_0 * decay_rate^k
```

### Connection to Label Smoothing

Self-distillation has a deep connection to label smoothing. Standard label smoothing replaces hard targets with:

```
y_smooth(c) = (1 - epsilon) * y_hard(c) + epsilon / C
```

where epsilon is the smoothing parameter and C is the number of classes. This is a fixed, uniform smoothing. Self-distillation achieves a data-dependent, non-uniform smoothing where the smoothing distribution comes from the model's own learned representations. This adaptive smoothing is generally more effective because it reflects the actual structure in the data rather than applying uniform uncertainty.

## 3. Self-Distillation Variants

### Born-Again Networks (BAN)

The original BAN approach trains each generation from scratch with randomly initialized weights. The only signal passed between generations is the soft target distribution. This ensures that each generation is not simply memorizing the previous model's weights but genuinely learning from the enriched supervision signal.

Key properties:
- Architecture remains identical across generations
- Weights are re-initialized each generation
- Soft targets are pre-computed and fixed during training
- Performance typically improves for 2-4 generations before saturating

### Be Your Own Teacher (BYOT)

The BYOT variant introduces auxiliary classifiers at intermediate layers of the network. Each intermediate layer produces its own predictions, and these predictions serve as additional soft targets for the final layer. This creates a form of internal self-distillation within a single training run:

- Shallow layers act as "teachers" for deeper layers
- Multiple loss terms are computed at different network depths
- The final loss is a weighted sum of all layer-wise losses
- This approach can be viewed as a form of deep supervision combined with self-distillation

In trading applications, BYOT is particularly useful because it provides multiple views of the data at different abstraction levels, which can capture both short-term patterns (shallow layers) and longer-term regime structures (deeper layers).

### Snapshot Distillation

Snapshot distillation leverages model checkpoints taken during training rather than training separate generations. By saving the model at regular intervals (snapshots), we create an implicit ensemble whose averaged predictions serve as soft targets for continued training:

1. Train the model and save snapshots at epochs E_1, E_2, ..., E_n
2. Average the predictions across all snapshots to create soft targets
3. Continue training using these averaged soft targets

This approach is computationally cheaper than BAN because it requires only a single extended training run rather than multiple full training cycles. The snapshots naturally capture different aspects of the data (early snapshots may capture simpler patterns while later ones capture complex interactions), creating a diverse ensemble for self-distillation.

## 4. Trading Applications

### Iterative Model Improvement Without Additional Data

In trading, acquiring labeled data is expensive and often subjective. Self-distillation allows us to extract more value from existing labeled data by iteratively refining the model. Each generation of self-distillation effectively increases the "effective" size of the training set by providing richer supervision signals.

Consider a market regime classifier trained on 1000 labeled candles. The hard labels are binary decisions (bull/bear/sideways), but the model's soft predictions after training encode probability distributions that capture the ambiguity in each sample. A candle that occurred during a regime transition might have a hard label of "bull" but a soft prediction of [0.55, 0.30, 0.15], reflecting the transitional nature of the market at that point. Training the next generation on these soft targets preserves and refines this nuance.

### Regularization Through Self-Distillation

Self-distillation acts as a powerful regularizer for trading models. Financial data is notoriously noisy, and models trained with hard labels tend to overfit to noise in the labels. Soft targets from self-distillation provide a natural smoothing effect that:

- Prevents the model from becoming overconfident on noisy samples
- Encourages the model to learn generalizable patterns rather than memorizing specific instances
- Reduces the impact of mislabeled data points (a mislabeled sample will still receive a soft target that partially reflects the true class)

This regularization effect is particularly valuable in low-data regimes common in trading, where we might have only a few hundred labeled regime transitions.

### Ensemble-Free Uncertainty Estimation

Traditional uncertainty estimation in trading requires maintaining multiple models (ensembles) or adding dropout at inference time (Monte Carlo dropout). Self-distillation provides an alternative: by tracking how predictions change across generations, we can estimate model uncertainty:

- **Stable predictions**: If a sample receives similar soft targets across generations 1-5, the model is confident about that prediction
- **Volatile predictions**: If soft targets shift significantly between generations, the model is uncertain
- **Generation disagreement**: The variance of predictions across generations provides a calibrated uncertainty measure

This is computationally free at inference time — we only need to deploy the final generation model, while uncertainty estimates are computed offline during the self-distillation process.

## 5. Multi-Generation Self-Distillation

### Training Over Multiple Generations

The multi-generation self-distillation process follows a specific protocol:

**Generation 0 (Baseline):**
- Train model M_0 on hard labels using standard cross-entropy
- Evaluate and record baseline accuracy

**Generation k (k >= 1):**
- Initialize a fresh model M_k with random weights
- Generate soft targets using M_{k-1} on the entire training set
- Train M_k using the combined loss: alpha * CE(hard) + (1-alpha) * KL(soft)
- Evaluate and record accuracy

**Hyperparameters across generations:**
- Temperature T: typically starts at 3-5 and decays toward 1
- Alpha (hard label weight): typically 0.3-0.5
- Learning rate: may be adjusted per generation based on convergence behavior

### Performance Saturation Analysis

Empirical studies show a characteristic pattern in self-distillation performance:

- **Generations 1-2**: Significant improvement (1-3% accuracy gain) as the model benefits from enriched supervision
- **Generations 3-4**: Diminishing returns, with smaller improvements
- **Generations 5+**: Performance saturates or may slightly degrade due to "echo chamber" effects where the model's biases become self-reinforcing

In trading applications, we typically observe optimal performance at generation 2-3. Beyond this point, the model may start to overfit to its own biases rather than continuing to improve. Monitoring validation set performance across generations is critical for selecting the optimal generation for deployment.

The saturation phenomenon can be mitigated by:
- Varying the temperature schedule across generations
- Adding stochastic noise to soft targets
- Using snapshot averaging rather than single-model targets
- Periodically injecting hard labels (curriculum-based alpha scheduling)

## 6. Implementation Walkthrough (Rust)

The implementation is structured around several core components that mirror the mathematical framework described above.

### Neural Network Architecture

We implement a simple feedforward neural network with configurable hidden layers. The key requirement is that the network must support cloning — each generation starts with a fresh copy of the architecture:

```rust
pub struct NeuralNetwork {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub layer_sizes: Vec<usize>,
}
```

The network supports forward propagation with temperature-scaled softmax output for soft target generation.

### Self-Distillation Training Loop

The core self-distillation loop implements the combined loss function:

```rust
pub fn self_distillation_step(
    student: &mut NeuralNetwork,
    teacher: &NeuralNetwork,
    input: &Array1<f64>,
    hard_label: usize,
    temperature: f64,
    alpha: f64,
    learning_rate: f64,
) -> f64
```

This function:
1. Forward-passes the input through both teacher (with temperature) and student
2. Computes the KL divergence between teacher and student soft outputs
3. Computes the cross-entropy between hard labels and student predictions
4. Combines losses with alpha weighting
5. Performs gradient descent on the student

### Multi-Generation Training

The `SelfDistillationTrainer` struct manages the full multi-generation pipeline:

```rust
pub struct SelfDistillationTrainer {
    pub layer_sizes: Vec<usize>,
    pub num_generations: usize,
    pub temperature: f64,
    pub alpha: f64,
    pub learning_rate: f64,
    pub epochs_per_generation: usize,
}
```

It tracks accuracy per generation, maintains generation history, and supports both BAN-style (fresh initialization) and snapshot-based self-distillation.

### Snapshot Distillation

The snapshot-based variant saves model checkpoints during training and averages their predictions:

```rust
pub struct SnapshotDistiller {
    pub snapshots: Vec<NeuralNetwork>,
    pub snapshot_interval: usize,
}
```

This provides a computationally efficient alternative to full multi-generation training.

## 7. Bybit Data Integration

The implementation includes a complete Bybit API integration for fetching real market data. The `fetch_bybit_klines` function retrieves candlestick data for any supported trading pair:

```rust
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<KlineData>>
```

Features are extracted from raw kline data and normalized to the [0, 1] range:
- **Return**: (close - open) / open, normalized
- **Volatility**: (high - low) / open, normalized
- **Volume change**: current volume relative to average, normalized

Market regime labels are assigned based on return magnitude and volatility levels:
- **Bull (0)**: Positive returns with moderate volatility
- **Bear (1)**: Negative returns with elevated volatility
- **Sideways (2)**: Near-zero returns

The trading example demonstrates the complete pipeline: fetching BTCUSDT data from Bybit, training an initial model, running 5 generations of self-distillation, and comparing accuracy across generations against a single-training baseline.

## 8. Key Takeaways

1. **Self-distillation eliminates the need for external teachers.** A model can improve itself by training successive generations on its own soft predictions, making it ideal for trading systems where simplicity matters.

2. **Soft targets encode richer information than hard labels.** The probability distributions produced by a trained model capture inter-class relationships and sample-level uncertainty that hard labels destroy.

3. **Temperature is the critical hyperparameter.** Higher temperatures produce softer distributions that emphasize class relationships, while lower temperatures focus on sharpening predictions. A decaying temperature schedule across generations provides the best of both worlds.

4. **Performance saturates after 2-4 generations.** Beyond this point, the model risks reinforcing its own biases. Validation monitoring is essential for selecting the optimal generation.

5. **Self-distillation provides implicit regularization.** The soft target smoothing effect naturally prevents overfitting, which is particularly valuable in the low-data, high-noise regime characteristic of financial markets.

6. **Multiple variants serve different needs.** BAN for maximum improvement, BYOT for single-training-run efficiency, and snapshot distillation for computational savings.

7. **Uncertainty estimation comes for free.** Tracking prediction stability across generations provides calibrated uncertainty estimates without maintaining ensembles at inference time.

8. **Rust implementation enables production deployment.** The zero-copy, low-latency characteristics of Rust make it suitable for real-time trading systems where self-distillation models must generate predictions with minimal overhead.
