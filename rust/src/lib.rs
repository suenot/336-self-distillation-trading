//! Self-Distillation Trading Library
//!
//! Implements self-distillation techniques for trading model improvement:
//! - Neural network with clone capability
//! - Born-Again Networks (BAN) style multi-generation self-distillation
//! - Soft target generation with temperature scaling
//! - KL divergence + cross-entropy combined loss
//! - Snapshot-based self-distillation
//! - Bybit API data fetching

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Neural Network
// ---------------------------------------------------------------------------

/// A simple feedforward neural network with clone capability.
/// Supports forward propagation with temperature-scaled softmax output.
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub layer_sizes: Vec<usize>,
}

impl NeuralNetwork {
    /// Create a new neural network with random weights.
    /// `layer_sizes` specifies the size of each layer including input and output.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];
            // Xavier initialization
            let scale = (2.0 / (cols + rows) as f64).sqrt();
            let w = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale));
            let b = Array1::zeros(rows);
            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    /// Number of input features.
    pub fn input_size(&self) -> usize {
        self.layer_sizes[0]
    }

    /// Number of output classes.
    pub fn output_size(&self) -> usize {
        *self.layer_sizes.last().unwrap()
    }

    /// Forward pass returning logits (pre-softmax output).
    pub fn forward_logits(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut activation = input.clone();

        for i in 0..self.weights.len() {
            let z = self.weights[i].dot(&activation) + &self.biases[i];
            if i < self.weights.len() - 1 {
                // ReLU for hidden layers
                activation = z.mapv(|x| x.max(0.0));
            } else {
                // Linear output (logits)
                activation = z;
            }
        }

        activation
    }

    /// Forward pass with softmax output at given temperature.
    pub fn forward_softmax(&self, input: &Array1<f64>, temperature: f64) -> Array1<f64> {
        let logits = self.forward_logits(input);
        softmax_with_temperature(&logits, temperature)
    }

    /// Forward pass with standard softmax (temperature = 1.0).
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward_softmax(input, 1.0)
    }

    /// Return predicted class (argmax).
    pub fn predict_class(&self, input: &Array1<f64>) -> usize {
        let probs = self.predict(input);
        argmax(&probs)
    }

    /// Compute accuracy on a dataset.
    pub fn accuracy(&self, inputs: &[Array1<f64>], labels: &[usize]) -> f64 {
        if inputs.is_empty() {
            return 0.0;
        }
        let correct = inputs
            .iter()
            .zip(labels.iter())
            .filter(|(x, &y)| self.predict_class(x) == y)
            .count();
        correct as f64 / inputs.len() as f64
    }

    /// Train one step with standard cross-entropy on hard labels.
    /// Returns the loss value.
    pub fn train_step_hard(
        &mut self,
        input: &Array1<f64>,
        label: usize,
        learning_rate: f64,
    ) -> f64 {
        let probs = self.predict(input);
        let loss = cross_entropy_loss(&probs, label);

        // Compute gradient of cross-entropy w.r.t. logits (softmax - one_hot)
        let mut grad_output = probs.clone();
        grad_output[label] -= 1.0;

        self.backward(input, &grad_output, learning_rate);
        loss
    }

    /// Backward pass and weight update via gradient descent.
    /// `grad_output` is the gradient of the loss w.r.t. the output logits.
    fn backward(
        &mut self,
        input: &Array1<f64>,
        grad_output: &Array1<f64>,
        learning_rate: f64,
    ) {
        // Forward pass to store activations
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();
        let mut a = input.clone();

        for i in 0..self.weights.len() {
            let z = self.weights[i].dot(&a) + &self.biases[i];
            pre_activations.push(z.clone());
            if i < self.weights.len() - 1 {
                a = z.mapv(|x| x.max(0.0));
            } else {
                a = z;
            }
            activations.push(a.clone());
        }

        // Backward pass
        let mut delta = grad_output.clone();

        for i in (0..self.weights.len()).rev() {
            // Gradient w.r.t. weights and biases
            let a_prev = &activations[i];
            let grad_w = outer_product(&delta, a_prev);
            let grad_b = delta.clone();

            // Update weights and biases
            self.weights[i] = &self.weights[i] - &(grad_w * learning_rate);
            self.biases[i] = &self.biases[i] - &(grad_b * learning_rate);

            // Propagate gradient to previous layer
            if i > 0 {
                let delta_prev = self.weights[i].t().dot(&delta);
                // ReLU derivative
                delta = Array1::from_shape_fn(delta_prev.len(), |j| {
                    if pre_activations[i - 1][j] > 0.0 {
                        delta_prev[j]
                    } else {
                        0.0
                    }
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Self-Distillation Training
// ---------------------------------------------------------------------------

/// Perform one self-distillation training step.
///
/// The student learns from both hard labels (cross-entropy) and
/// soft targets from the teacher (KL divergence).
///
/// Returns the combined loss value.
pub fn self_distillation_step(
    student: &mut NeuralNetwork,
    teacher: &NeuralNetwork,
    input: &Array1<f64>,
    hard_label: usize,
    temperature: f64,
    alpha: f64,
    learning_rate: f64,
) -> f64 {
    // Teacher soft targets (with temperature)
    let teacher_soft = teacher.forward_softmax(input, temperature);

    // Student predictions
    let student_probs = student.predict(input);
    let student_soft = student.forward_softmax(input, temperature);

    // Cross-entropy loss with hard labels
    let ce_loss = cross_entropy_loss(&student_probs, hard_label);

    // KL divergence loss between teacher and student soft targets
    let kl_loss = kl_divergence(&teacher_soft, &student_soft);

    // Combined loss
    let total_loss = alpha * ce_loss + (1.0 - alpha) * temperature * temperature * kl_loss;

    // Gradient: combination of CE gradient and KL gradient
    let mut grad_output = student_probs.clone();
    grad_output[hard_label] -= 1.0;
    let grad_ce = grad_output;

    // KL gradient w.r.t. student logits: (student_soft - teacher_soft) * T
    let grad_kl = (&student_soft - &teacher_soft) * temperature;

    // Combined gradient
    let grad_combined = grad_ce * alpha + grad_kl * (1.0 - alpha) * temperature;

    student.backward(input, &grad_combined, learning_rate);

    total_loss
}

/// Train a full epoch of self-distillation.
/// Returns the average loss.
pub fn self_distillation_epoch(
    student: &mut NeuralNetwork,
    teacher: &NeuralNetwork,
    inputs: &[Array1<f64>],
    labels: &[usize],
    temperature: f64,
    alpha: f64,
    learning_rate: f64,
) -> f64 {
    if inputs.is_empty() {
        return 0.0;
    }
    let mut total_loss = 0.0;
    for (input, &label) in inputs.iter().zip(labels.iter()) {
        total_loss += self_distillation_step(
            student,
            teacher,
            input,
            label,
            temperature,
            alpha,
            learning_rate,
        );
    }
    total_loss / inputs.len() as f64
}

// ---------------------------------------------------------------------------
// Multi-Generation Self-Distillation (BAN Style)
// ---------------------------------------------------------------------------

/// Result of a single generation of self-distillation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub generation: usize,
    pub accuracy: f64,
    pub avg_loss: f64,
}

/// Self-distillation trainer that manages multi-generation training.
pub struct SelfDistillationTrainer {
    pub layer_sizes: Vec<usize>,
    pub num_generations: usize,
    pub temperature: f64,
    pub alpha: f64,
    pub learning_rate: f64,
    pub epochs_per_generation: usize,
    pub temperature_decay: f64,
}

impl SelfDistillationTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(
        layer_sizes: Vec<usize>,
        num_generations: usize,
        temperature: f64,
        alpha: f64,
        learning_rate: f64,
        epochs_per_generation: usize,
    ) -> Self {
        Self {
            layer_sizes,
            num_generations,
            temperature,
            alpha,
            learning_rate,
            epochs_per_generation,
            temperature_decay: 0.8,
        }
    }

    /// Set the temperature decay rate between generations.
    pub fn with_temperature_decay(mut self, decay: f64) -> Self {
        self.temperature_decay = decay;
        self
    }

    /// Run multi-generation self-distillation (Born-Again Networks style).
    ///
    /// Returns a vector of generation results tracking accuracy improvement.
    pub fn train(
        &self,
        inputs: &[Array1<f64>],
        labels: &[usize],
    ) -> Vec<GenerationResult> {
        let mut results = Vec::new();

        // Generation 0: train on hard labels only
        let mut teacher = NeuralNetwork::new(&self.layer_sizes);
        let mut avg_loss = 0.0;
        for _epoch in 0..self.epochs_per_generation {
            let mut epoch_loss = 0.0;
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                epoch_loss += teacher.train_step_hard(input, label, self.learning_rate);
            }
            avg_loss = epoch_loss / inputs.len() as f64;
        }

        let acc = teacher.accuracy(inputs, labels);
        results.push(GenerationResult {
            generation: 0,
            accuracy: acc,
            avg_loss,
        });

        // Generations 1..N: self-distillation
        for gen in 1..=self.num_generations {
            let temp = self.temperature * self.temperature_decay.powi((gen - 1) as i32);
            let mut student = NeuralNetwork::new(&self.layer_sizes);

            for _epoch in 0..self.epochs_per_generation {
                avg_loss = self_distillation_epoch(
                    &mut student,
                    &teacher,
                    inputs,
                    labels,
                    temp,
                    self.alpha,
                    self.learning_rate,
                );
            }

            let acc = student.accuracy(inputs, labels);
            results.push(GenerationResult {
                generation: gen,
                accuracy: acc,
                avg_loss,
            });

            // Student becomes the teacher for the next generation
            teacher = student;
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Snapshot-Based Self-Distillation
// ---------------------------------------------------------------------------

/// Snapshot distiller that saves model checkpoints during training
/// and uses averaged predictions as soft targets.
pub struct SnapshotDistiller {
    pub snapshots: Vec<NeuralNetwork>,
    pub snapshot_interval: usize,
}

impl SnapshotDistiller {
    /// Create a new snapshot distiller.
    pub fn new(snapshot_interval: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            snapshot_interval,
        }
    }

    /// Train a model with snapshot collection.
    /// Returns the trained model and populates the snapshots.
    pub fn train_with_snapshots(
        &mut self,
        layer_sizes: &[usize],
        inputs: &[Array1<f64>],
        labels: &[usize],
        total_epochs: usize,
        learning_rate: f64,
    ) -> NeuralNetwork {
        let mut model = NeuralNetwork::new(layer_sizes);
        self.snapshots.clear();

        for epoch in 0..total_epochs {
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                model.train_step_hard(input, label, learning_rate);
            }

            // Save snapshot at regular intervals
            if (epoch + 1) % self.snapshot_interval == 0 {
                self.snapshots.push(model.clone());
            }
        }

        model
    }

    /// Generate averaged soft targets from all snapshots for a given input.
    pub fn averaged_soft_targets(&self, input: &Array1<f64>, temperature: f64) -> Option<Array1<f64>> {
        if self.snapshots.is_empty() {
            return None;
        }

        let n = self.snapshots.len() as f64;
        let first = self.snapshots[0].forward_softmax(input, temperature);
        let mut avg = first;

        for snapshot in &self.snapshots[1..] {
            let soft = snapshot.forward_softmax(input, temperature);
            avg = avg + soft;
        }

        Some(avg / n)
    }

    /// Continue training the model using averaged snapshot predictions as soft targets.
    pub fn distill_from_snapshots(
        &self,
        model: &mut NeuralNetwork,
        inputs: &[Array1<f64>],
        labels: &[usize],
        epochs: usize,
        temperature: f64,
        alpha: f64,
        learning_rate: f64,
    ) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }

        let mut avg_loss = 0.0;
        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                // Get averaged soft targets
                let teacher_soft = self
                    .averaged_soft_targets(input, temperature)
                    .unwrap();

                let student_probs = model.predict(input);
                let student_soft = model.forward_softmax(input, temperature);

                let ce_loss = cross_entropy_loss(&student_probs, label);
                let kl_loss = kl_divergence(&teacher_soft, &student_soft);
                let total_loss =
                    alpha * ce_loss + (1.0 - alpha) * temperature * temperature * kl_loss;

                // Gradient
                let mut grad_ce = student_probs.clone();
                grad_ce[label] -= 1.0;
                let grad_kl = (&student_soft - &teacher_soft) * temperature;
                let grad = grad_ce * alpha + grad_kl * (1.0 - alpha) * temperature;

                model.backward(input, &grad, learning_rate);
                epoch_loss += total_loss;
            }
            avg_loss = epoch_loss / inputs.len() as f64;
        }
        avg_loss
    }

    /// Number of snapshots collected.
    pub fn num_snapshots(&self) -> usize {
        self.snapshots.len()
    }
}

// ---------------------------------------------------------------------------
// Soft Target Generation
// ---------------------------------------------------------------------------

/// Generate soft targets from a model for the entire dataset.
pub fn generate_soft_targets(
    model: &NeuralNetwork,
    inputs: &[Array1<f64>],
    temperature: f64,
) -> Vec<Array1<f64>> {
    inputs
        .iter()
        .map(|input| model.forward_softmax(input, temperature))
        .collect()
}

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------

/// Cross-entropy loss for a single sample.
pub fn cross_entropy_loss(probs: &Array1<f64>, label: usize) -> f64 {
    let p = probs[label].max(1e-15);
    -p.ln()
}

/// KL divergence: D_KL(p || q) = sum_i p_i * log(p_i / q_i)
pub fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    let mut kl = 0.0;
    for i in 0..p.len() {
        if p[i] > 1e-15 {
            let q_i = q[i].max(1e-15);
            kl += p[i] * (p[i] / q_i).ln();
        }
    }
    kl.max(0.0)
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Softmax with temperature scaling.
pub fn softmax_with_temperature(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let t = temperature.max(1e-10);
    let scaled = logits / t;
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Array1<f64> = scaled.mapv(|x| (x - max_val).exp());
    let sum: f64 = exps.sum();
    if sum < 1e-15 {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    } else {
        exps / sum
    }
}

/// Standard softmax.
pub fn softmax(scores: &Array1<f64>) -> Array1<f64> {
    softmax_with_temperature(scores, 1.0)
}

/// Argmax of an array.
pub fn argmax(a: &Array1<f64>) -> usize {
    a.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Outer product of two vectors.
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

// ---------------------------------------------------------------------------
// Bybit API Data Fetching
// ---------------------------------------------------------------------------

/// Raw kline data from Bybit API.
#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit REST API.
///
/// * `symbol` - Trading pair, e.g. "BTCUSDT"
/// * `interval` - Candle interval, e.g. "60" for 1 hour
/// * `limit` - Number of candles to fetch (max 200)
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: ret_code={}", resp.ret_code));
    }

    let mut klines = Vec::new();
    for entry in &resp.result.list {
        if entry.len() < 6 {
            continue;
        }
        klines.push(KlineData {
            timestamp: entry[0].parse().unwrap_or(0),
            open: entry[1].parse().unwrap_or(0.0),
            high: entry[2].parse().unwrap_or(0.0),
            low: entry[3].parse().unwrap_or(0.0),
            close: entry[4].parse().unwrap_or(0.0),
            volume: entry[5].parse().unwrap_or(0.0),
        });
    }

    // Bybit returns newest first; reverse so oldest is first.
    klines.reverse();

    Ok(klines)
}

/// Extract normalized features from kline data.
/// Returns vectors of: [return, volatility, volume_change] each in [0, 1].
pub fn extract_features(klines: &[KlineData]) -> Vec<[f64; 3]> {
    if klines.is_empty() {
        return vec![];
    }

    let avg_volume: f64 = klines.iter().map(|k| k.volume).sum::<f64>() / klines.len() as f64;

    klines
        .iter()
        .map(|k| {
            let ret = if k.open != 0.0 {
                ((k.close - k.open) / k.open).clamp(-0.1, 0.1) / 0.2 + 0.5
            } else {
                0.5
            };
            let vol = if k.open != 0.0 {
                ((k.high - k.low) / k.open).clamp(0.0, 0.2) / 0.2
            } else {
                0.0
            };
            let vol_change = if avg_volume > 0.0 {
                (k.volume / avg_volume).clamp(0.0, 3.0) / 3.0
            } else {
                0.0
            };
            [ret, vol, vol_change]
        })
        .collect()
}

/// Simple market regime labeling based on returns and volatility.
/// 0 = bull, 1 = bear, 2 = sideways
pub fn label_regime(features: &[f64; 3]) -> usize {
    let ret = features[0]; // 0.5 = zero return
    let vol = features[1];

    if ret > 0.55 && vol < 0.5 {
        0 // bull
    } else if ret < 0.45 && vol > 0.3 {
        1 // bear
    } else {
        2 // sideways
    }
}

/// Convert raw features to neural network input vector.
pub fn features_to_input(features: &[f64; 3]) -> Array1<f64> {
    Array1::from_vec(features.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(&[3, 8, 3]);
        assert_eq!(nn.input_size(), 3);
        assert_eq!(nn.output_size(), 3);
        assert_eq!(nn.weights.len(), 2);
        assert_eq!(nn.biases.len(), 2);
    }

    #[test]
    fn test_neural_network_forward() {
        let nn = NeuralNetwork::new(&[3, 8, 3]);
        let input = Array1::from_vec(vec![0.5, 0.3, 0.7]);
        let probs = nn.predict(&input);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Softmax output should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_neural_network_clone() {
        let nn = NeuralNetwork::new(&[3, 8, 3]);
        let nn_clone = nn.clone();
        let input = Array1::from_vec(vec![0.5, 0.3, 0.7]);
        let probs1 = nn.predict(&input);
        let probs2 = nn_clone.predict(&input);
        for i in 0..3 {
            assert!(
                (probs1[i] - probs2[i]).abs() < 1e-15,
                "Cloned network should produce identical output"
            );
        }
    }

    #[test]
    fn test_softmax_with_temperature() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // T=1: standard softmax
        let p1 = softmax_with_temperature(&logits, 1.0);
        assert!((p1.sum() - 1.0).abs() < 1e-10);
        assert!(p1[2] > p1[1] && p1[1] > p1[0]);

        // T=10: much softer distribution
        let p10 = softmax_with_temperature(&logits, 10.0);
        assert!((p10.sum() - 1.0).abs() < 1e-10);
        // Higher temperature -> more uniform distribution
        let range_1 = p1[2] - p1[0];
        let range_10 = p10[2] - p10[0];
        assert!(
            range_10 < range_1,
            "Higher temperature should produce softer distribution"
        );
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = Array1::from_vec(vec![0.3, 0.5, 0.2]);
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "KL divergence of identical distributions should be 0");
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = Array1::from_vec(vec![0.9, 0.05, 0.05]);
        let q = Array1::from_vec(vec![0.33, 0.34, 0.33]);
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL divergence of different distributions should be positive");
    }

    #[test]
    fn test_cross_entropy_loss() {
        let probs = Array1::from_vec(vec![0.7, 0.2, 0.1]);
        let loss = cross_entropy_loss(&probs, 0);
        assert!(loss > 0.0, "Cross-entropy loss should be positive");
        assert!(loss < 1.0, "Loss for 0.7 probability should be small");

        // Higher probability -> lower loss
        let probs2 = Array1::from_vec(vec![0.9, 0.05, 0.05]);
        let loss2 = cross_entropy_loss(&probs2, 0);
        assert!(loss2 < loss, "Higher probability should give lower loss");
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut nn = NeuralNetwork::new(&[3, 16, 3]);
        let inputs = vec![
            Array1::from_vec(vec![0.8, 0.1, 0.5]),
            Array1::from_vec(vec![0.2, 0.7, 0.6]),
            Array1::from_vec(vec![0.5, 0.4, 0.3]),
        ];
        let labels = vec![0, 1, 2];

        let initial_loss: f64 = inputs
            .iter()
            .zip(labels.iter())
            .map(|(x, &y)| cross_entropy_loss(&nn.predict(x), y))
            .sum::<f64>()
            / 3.0;

        for _ in 0..100 {
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                nn.train_step_hard(input, label, 0.01);
            }
        }

        let final_loss: f64 = inputs
            .iter()
            .zip(labels.iter())
            .map(|(x, &y)| cross_entropy_loss(&nn.predict(x), y))
            .sum::<f64>()
            / 3.0;

        assert!(
            final_loss < initial_loss,
            "Training should reduce loss: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_self_distillation_step() {
        let teacher = NeuralNetwork::new(&[3, 8, 3]);
        let mut student = NeuralNetwork::new(&[3, 8, 3]);
        let input = Array1::from_vec(vec![0.5, 0.3, 0.7]);

        let loss = self_distillation_step(
            &mut student,
            &teacher,
            &input,
            0,
            3.0,
            0.5,
            0.01,
        );
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_self_distillation_epoch() {
        let teacher = NeuralNetwork::new(&[3, 8, 3]);
        let mut student = NeuralNetwork::new(&[3, 8, 3]);
        let inputs = vec![
            Array1::from_vec(vec![0.8, 0.1, 0.5]),
            Array1::from_vec(vec![0.2, 0.7, 0.6]),
        ];
        let labels = vec![0, 1];

        let loss = self_distillation_epoch(
            &mut student,
            &teacher,
            &inputs,
            &labels,
            3.0,
            0.5,
            0.01,
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn test_multi_generation_training() {
        let trainer = SelfDistillationTrainer::new(
            vec![3, 16, 3],
            3,    // 3 generations
            3.0,  // temperature
            0.5,  // alpha
            0.01, // learning rate
            50,   // epochs per generation
        );

        let inputs = vec![
            Array1::from_vec(vec![0.8, 0.1, 0.5]),
            Array1::from_vec(vec![0.2, 0.7, 0.6]),
            Array1::from_vec(vec![0.5, 0.4, 0.3]),
            Array1::from_vec(vec![0.9, 0.2, 0.4]),
            Array1::from_vec(vec![0.1, 0.8, 0.7]),
            Array1::from_vec(vec![0.4, 0.5, 0.2]),
        ];
        let labels = vec![0, 1, 2, 0, 1, 2];

        let results = trainer.train(&inputs, &labels);
        assert_eq!(results.len(), 4); // gen 0 + 3 generations
        assert_eq!(results[0].generation, 0);
        assert_eq!(results[3].generation, 3);

        // All accuracies should be valid
        for r in &results {
            assert!(r.accuracy >= 0.0 && r.accuracy <= 1.0);
            assert!(r.avg_loss.is_finite());
        }
    }

    #[test]
    fn test_snapshot_distiller() {
        let mut distiller = SnapshotDistiller::new(5);

        let inputs = vec![
            Array1::from_vec(vec![0.8, 0.1, 0.5]),
            Array1::from_vec(vec![0.2, 0.7, 0.6]),
            Array1::from_vec(vec![0.5, 0.4, 0.3]),
        ];
        let labels = vec![0, 1, 2];

        let _model = distiller.train_with_snapshots(
            &[3, 8, 3],
            &inputs,
            &labels,
            20,
            0.01,
        );

        assert_eq!(distiller.num_snapshots(), 4); // 20 / 5 = 4 snapshots

        // Test averaged soft targets
        let soft = distiller.averaged_soft_targets(&inputs[0], 3.0);
        assert!(soft.is_some());
        let soft = soft.unwrap();
        assert_eq!(soft.len(), 3);
        assert!((soft.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_distillation() {
        let mut distiller = SnapshotDistiller::new(10);

        let inputs = vec![
            Array1::from_vec(vec![0.8, 0.1, 0.5]),
            Array1::from_vec(vec![0.2, 0.7, 0.6]),
            Array1::from_vec(vec![0.5, 0.4, 0.3]),
        ];
        let labels = vec![0, 1, 2];

        let mut model = distiller.train_with_snapshots(
            &[3, 8, 3],
            &inputs,
            &labels,
            30,
            0.01,
        );

        let loss = distiller.distill_from_snapshots(
            &mut model,
            &inputs,
            &labels,
            10,
            3.0,
            0.5,
            0.01,
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn test_generate_soft_targets() {
        let model = NeuralNetwork::new(&[3, 8, 3]);
        let inputs = vec![
            Array1::from_vec(vec![0.5, 0.3, 0.7]),
            Array1::from_vec(vec![0.1, 0.9, 0.2]),
        ];

        let soft = generate_soft_targets(&model, &inputs, 3.0);
        assert_eq!(soft.len(), 2);
        for s in &soft {
            assert_eq!(s.len(), 3);
            assert!((s.sum() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_temperature_decay() {
        let trainer = SelfDistillationTrainer::new(
            vec![3, 8, 3],
            3,
            5.0,
            0.5,
            0.01,
            10,
        )
        .with_temperature_decay(0.5);

        assert!((trainer.temperature - 5.0).abs() < 1e-10);
        assert!((trainer.temperature_decay - 0.5).abs() < 1e-10);

        // Verify temperature schedule
        let t0 = trainer.temperature;
        let t1 = t0 * trainer.temperature_decay;
        let t2 = t0 * trainer.temperature_decay.powi(2);
        assert!((t0 - 5.0).abs() < 1e-10);
        assert!((t1 - 2.5).abs() < 1e-10);
        assert!((t2 - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_extract_features() {
        let klines = vec![
            KlineData {
                timestamp: 1,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 103.0,
                volume: 1000.0,
            },
            KlineData {
                timestamp: 2,
                open: 103.0,
                high: 107.0,
                low: 100.0,
                close: 101.0,
                volume: 1200.0,
            },
        ];
        let features = extract_features(&klines);
        assert_eq!(features.len(), 2);
        for f in &features {
            for &v in f {
                assert!(v >= 0.0 && v <= 1.0, "Feature {} out of [0,1]", v);
            }
        }
    }

    #[test]
    fn test_label_regime() {
        assert_eq!(label_regime(&[0.7, 0.2, 0.5]), 0); // bull
        assert_eq!(label_regime(&[0.3, 0.5, 0.5]), 1); // bear
        assert_eq!(label_regime(&[0.5, 0.3, 0.5]), 2); // sideways
    }

    #[test]
    fn test_features_to_input() {
        let features = [0.7, 0.3, 0.5];
        let input = features_to_input(&features);
        assert_eq!(input.len(), 3);
        assert!((input[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_argmax() {
        let a = Array1::from_vec(vec![0.1, 0.7, 0.2]);
        assert_eq!(argmax(&a), 1);

        let b = Array1::from_vec(vec![0.9, 0.05, 0.05]);
        assert_eq!(argmax(&b), 0);
    }

    #[test]
    fn test_accuracy() {
        let mut nn = NeuralNetwork::new(&[3, 16, 3]);
        let inputs = vec![
            Array1::from_vec(vec![0.9, 0.1, 0.1]),
            Array1::from_vec(vec![0.1, 0.9, 0.1]),
        ];
        let labels = vec![0, 1];

        // Train to get reasonable accuracy
        for _ in 0..200 {
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                nn.train_step_hard(input, label, 0.01);
            }
        }

        let acc = nn.accuracy(&inputs, &labels);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
