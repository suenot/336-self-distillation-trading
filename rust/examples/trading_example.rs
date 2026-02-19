//! Trading Example: Self-Distillation for Market Regime Detection
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT data from Bybit
//! 2. Training an initial model on market regime classification
//! 3. Running 5 generations of self-distillation (BAN style)
//! 4. Showing accuracy improvement across generations
//! 5. Comparing with single-training baseline

use anyhow::Result;
use self_distillation_trading::*;

/// Number of candles to fetch
const NUM_CANDLES: usize = 200;

/// Number of self-distillation generations
const NUM_GENERATIONS: usize = 5;

/// Hidden layer size
const HIDDEN_SIZE: usize = 32;

/// Number of output classes (bull, bear, sideways)
const NUM_CLASSES: usize = 3;

/// Number of input features (return, volatility, volume_change)
const NUM_FEATURES: usize = 3;

/// Learning rate
const LEARNING_RATE: f64 = 0.005;

/// Epochs per generation
const EPOCHS_PER_GEN: usize = 100;

/// Temperature for self-distillation
const TEMPERATURE: f64 = 4.0;

/// Alpha (weight of hard labels vs soft targets)
const ALPHA: f64 = 0.4;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Self-Distillation Trading: Market Regime Detection ===\n");

    // Step 1: Fetch data from Bybit
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let features = match fetch_bybit_klines("BTCUSDT", "60", NUM_CANDLES).await {
        Ok(klines) => {
            println!("  Fetched {} candles from Bybit", klines.len());
            extract_features(&klines)
        }
        Err(e) => {
            println!("  Error fetching data: {}. Using synthetic data.", e);
            generate_synthetic_features(NUM_CANDLES)
        }
    };

    // Step 2: Prepare training data
    println!("\nStep 2: Preparing training data...");
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for f in &features {
        inputs.push(features_to_input(f));
        labels.push(label_regime(f));
    }

    let regime_counts = [
        labels.iter().filter(|&&l| l == 0).count(),
        labels.iter().filter(|&&l| l == 1).count(),
        labels.iter().filter(|&&l| l == 2).count(),
    ];
    println!(
        "  Samples: {} (Bull: {}, Bear: {}, Sideways: {})",
        inputs.len(),
        regime_counts[0],
        regime_counts[1],
        regime_counts[2]
    );

    // Split into train/test (80/20)
    let split = (inputs.len() as f64 * 0.8) as usize;
    let train_inputs = &inputs[..split];
    let train_labels = &labels[..split];
    let test_inputs = &inputs[split..];
    let test_labels = &labels[split..];
    println!(
        "  Train: {}, Test: {}",
        train_inputs.len(),
        test_inputs.len()
    );

    // Step 3: Train baseline (single training, no distillation)
    println!("\nStep 3: Training baseline model (no self-distillation)...");
    let mut baseline = NeuralNetwork::new(&[NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES]);
    for epoch in 0..EPOCHS_PER_GEN * (NUM_GENERATIONS + 1) {
        for (input, &label) in train_inputs.iter().zip(train_labels.iter()) {
            baseline.train_step_hard(input, label, LEARNING_RATE);
        }
        if (epoch + 1) % 100 == 0 {
            let train_acc = baseline.accuracy(train_inputs, train_labels);
            let test_acc = baseline.accuracy(test_inputs, test_labels);
            println!(
                "  Epoch {}: train_acc={:.1}%, test_acc={:.1}%",
                epoch + 1,
                train_acc * 100.0,
                test_acc * 100.0
            );
        }
    }
    let baseline_train_acc = baseline.accuracy(train_inputs, train_labels);
    let baseline_test_acc = baseline.accuracy(test_inputs, test_labels);
    println!(
        "  Baseline final: train_acc={:.1}%, test_acc={:.1}%",
        baseline_train_acc * 100.0,
        baseline_test_acc * 100.0
    );

    // Step 4: Multi-generation self-distillation
    println!("\nStep 4: Running {} generations of self-distillation...", NUM_GENERATIONS);
    println!(
        "  Config: temperature={}, alpha={}, lr={}, epochs/gen={}",
        TEMPERATURE, ALPHA, LEARNING_RATE, EPOCHS_PER_GEN
    );

    let trainer = SelfDistillationTrainer::new(
        vec![NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES],
        NUM_GENERATIONS,
        TEMPERATURE,
        ALPHA,
        LEARNING_RATE,
        EPOCHS_PER_GEN,
    )
    .with_temperature_decay(0.85);

    let results = trainer.train(train_inputs, train_labels);

    println!("\n  Generation Results (Training Set):");
    println!("  {:<12} {:<15} {:<15}", "Generation", "Accuracy", "Avg Loss");
    println!("  {}", "-".repeat(42));
    for r in &results {
        println!(
            "  {:<12} {:<15.1}% {:<15.6}",
            r.generation,
            r.accuracy * 100.0,
            r.avg_loss
        );
    }

    // Step 5: Evaluate best generation on test set
    println!("\nStep 5: Evaluating on test set...");

    // Re-run to get the actual models for test evaluation
    // Generation 0: baseline
    let mut teacher = NeuralNetwork::new(&[NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES]);
    for _epoch in 0..EPOCHS_PER_GEN {
        for (input, &label) in train_inputs.iter().zip(train_labels.iter()) {
            teacher.train_step_hard(input, label, LEARNING_RATE);
        }
    }
    let gen0_test_acc = teacher.accuracy(test_inputs, test_labels);

    println!("\n  Test Set Accuracy by Generation:");
    println!("  Generation 0: {:.1}%", gen0_test_acc * 100.0);

    let mut best_acc = gen0_test_acc;
    let mut best_gen = 0;

    for gen in 1..=NUM_GENERATIONS {
        let temp = TEMPERATURE * 0.85_f64.powi((gen - 1) as i32);
        let mut student = NeuralNetwork::new(&[NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES]);

        for _epoch in 0..EPOCHS_PER_GEN {
            self_distillation_epoch(
                &mut student,
                &teacher,
                train_inputs,
                train_labels,
                temp,
                ALPHA,
                LEARNING_RATE,
            );
        }

        let test_acc = student.accuracy(test_inputs, test_labels);
        println!(
            "  Generation {}: {:.1}% (T={:.2})",
            gen,
            test_acc * 100.0,
            temp
        );

        if test_acc > best_acc {
            best_acc = test_acc;
            best_gen = gen;
        }

        teacher = student;
    }

    // Step 6: Snapshot distillation comparison
    println!("\nStep 6: Snapshot distillation comparison...");
    let mut snapshot_distiller = SnapshotDistiller::new(20);
    let mut snapshot_model = snapshot_distiller.train_with_snapshots(
        &[NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES],
        train_inputs,
        train_labels,
        EPOCHS_PER_GEN,
        LEARNING_RATE,
    );

    let pre_distill_acc = snapshot_model.accuracy(test_inputs, test_labels);
    println!(
        "  Before snapshot distillation: test_acc={:.1}%",
        pre_distill_acc * 100.0
    );
    println!("  Snapshots collected: {}", snapshot_distiller.num_snapshots());

    snapshot_distiller.distill_from_snapshots(
        &mut snapshot_model,
        train_inputs,
        train_labels,
        50,
        3.0,
        0.4,
        LEARNING_RATE,
    );

    let post_distill_acc = snapshot_model.accuracy(test_inputs, test_labels);
    println!(
        "  After snapshot distillation: test_acc={:.1}%",
        post_distill_acc * 100.0
    );

    // Step 7: Summary
    println!("\n=== Summary ===");
    println!(
        "  Single training baseline:    test_acc={:.1}%",
        baseline_test_acc * 100.0
    );
    println!(
        "  Best BAN generation (gen {}): test_acc={:.1}%",
        best_gen,
        best_acc * 100.0
    );
    println!(
        "  Snapshot distillation:       test_acc={:.1}%",
        post_distill_acc * 100.0
    );

    if best_acc > baseline_test_acc {
        println!(
            "\n  Self-distillation improved test accuracy by {:.1} percentage points!",
            (best_acc - baseline_test_acc) * 100.0
        );
    } else {
        println!("\n  Baseline performed well. Self-distillation may help more with larger datasets.");
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic market features for demonstration when API is unavailable.
fn generate_synthetic_features(n: usize) -> Vec<[f64; 3]> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut features = Vec::with_capacity(n);

    let mut regime = 0;
    let mut regime_counter = 0;

    for _ in 0..n {
        regime_counter += 1;
        if regime_counter > 15 + rng.gen_range(0..10) {
            regime = rng.gen_range(0..3);
            regime_counter = 0;
        }

        let (ret, vol, vol_chg) = match regime {
            0 => (
                0.5 + rng.gen_range(0.05..0.15),
                rng.gen_range(0.1..0.3),
                rng.gen_range(0.3..0.6),
            ),
            1 => (
                0.5 - rng.gen_range(0.05..0.15),
                rng.gen_range(0.4..0.7),
                rng.gen_range(0.5..0.8),
            ),
            _ => (
                0.5 + rng.gen_range(-0.04..0.04),
                rng.gen_range(0.2..0.4),
                rng.gen_range(0.2..0.5),
            ),
        };

        features.push([ret, vol, vol_chg]);
    }

    features
}
