use crate::detectors::DetectorResult;
use serde::{Deserialize, Serialize};

/// Configuration du module d'apprentissage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerConfig {
    /// Taille de la fenêtre d'entraînement
    pub training_window_size: usize,
    /// Nombre d'époques d'entraînement
    pub epochs: usize,
    /// Taux d'apprentissage
    pub learning_rate: f64,
    /// Seuil d'anomalie (distance au modèle > threshold = anomalie)
    pub anomaly_threshold: f64,
    /// Nombre de features d'entrée
    pub input_features: usize,
    /// Taille de la couche cachée
    pub hidden_size: usize,
    /// Pourcentage de données normales pour l'entraînement
    pub normal_data_ratio: f64,
    /// EMA alpha pour la mise à jour du modèle
    pub ema_alpha: f64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            training_window_size: 1000,
            epochs: 50,
            learning_rate: 0.001,
            anomaly_threshold: 2.0,
            input_features: 10,
            hidden_size: 16,
            normal_data_ratio: 0.95,
            ema_alpha: 0.1,
        }
    }
}

/// Modèle d'anomalie basé sur un autoencodeur simple (MLP).
///
/// Entraîné sur du trafic normal, détecte les anomalies par la
/// distance de reconstruction: plus la reconstruction est mauvaise,
/// plus le trafic est suspect.
#[derive(Debug, Clone)]
pub struct AnomalyModel {
    pub config: LearnerConfig,
    /// Poids couche d'entrée -> cachée (input_size x hidden_size)
    w1: Vec<Vec<f64>>,
    /// Biais couche cachée
    b1: Vec<f64>,
    /// Poids couche cachée -> sortie (hidden_size x input_size)
    w2: Vec<Vec<f64>>,
    /// Biais couche de sortie
    b2: Vec<f64>,
    /// Seuil d'anomalie calibré
    calibrated_threshold: f64,
    /// Nombre d'échantillons vus
    sample_count: u64,
    /// MSE moyen sur les données d'entraînement
    training_mse: f64,
}

impl AnomalyModel {
    pub fn new(config: LearnerConfig) -> Self {
        let mut rng_state: u64 = 42;
        let mut rng = || -> f64 {
            // LCG simple pour init déterministe
            rng_state = rng_state.wrapping_add(6364136223846793005);
            ((rng_state >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        };

        // Xavier initialization
        let scale1 = (2.0 / config.input_features as f64).sqrt();
        let scale2 = (2.0 / config.hidden_size as f64).sqrt();

        let w1: Vec<Vec<f64>> = (0..config.input_features)
            .map(|_| (0..config.hidden_size).map(|_| rng() * scale1).collect())
            .collect();
        let b1 = vec![0.0; config.hidden_size];

        let w2: Vec<Vec<f64>> = (0..config.hidden_size)
            .map(|_| (0..config.input_features).map(|_| rng() * scale2).collect())
            .collect();
        let b2 = vec![0.0; config.input_features];

        Self {
            config,
            w1,
            b1,
            w2,
            b2,
            calibrated_threshold: 0.0,
            sample_count: 0,
            training_mse: 0.0,
        }
    }

    pub fn config(&self) -> &LearnerConfig {
        &self.config
    }

    /// Forward pass: encoder -> decoder.
    #[allow(clippy::needless_range_loop)] // dense matrix loops index w1/w2/b by position
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(input.len(), self.config.input_features);

        // Encoder: hidden = relu(W1 * input + b1)
        let mut hidden = vec![0.0f64; self.config.hidden_size];
        for j in 0..self.config.hidden_size
        {
            let mut sum = self.b1[j];
            for i in 0..self.config.input_features
            {
                sum += self.w1[i][j] * input[i];
            }
            // ReLU
            hidden[j] = sum.max(0.0);
        }

        // Decoder: output = W2 * hidden + b2
        let mut output = vec![0.0f64; self.config.input_features];
        for i in 0..self.config.input_features
        {
            let mut sum = self.b2[i];
            for j in 0..self.config.hidden_size
            {
                sum += self.w2[j][i] * hidden[j];
            }
            output[i] = sum;
        }

        (hidden, output)
    }

    /// MSE (Mean Squared Error) de reconstruction.
    fn reconstruction_mse(&self, input: &[f64], output: &[f64]) -> f64 {
        let n = input.len() as f64;
        input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n
    }

    /// Entraîner le modèle sur des données normales (autoencoding).
    #[allow(clippy::needless_range_loop)] // dense backprop loops index w1/w2/b by position
    pub fn train(&mut self, normal_data: &[Vec<f64>]) {
        if normal_data.is_empty()
        {
            return;
        }

        let lr = self.config.learning_rate;

        for _epoch in 0..self.config.epochs
        {
            let mut epoch_mse = 0.0;

            for sample in normal_data
            {
                let (hidden, output) = self.forward(sample);

                // Calcul de l'erreur
                let error: Vec<f64> = sample
                    .iter()
                    .zip(output.iter())
                    .map(|(a, b)| a - b)
                    .collect();

                epoch_mse += self.reconstruction_mse(sample, &output);

                // Backprop: update W2 (decoder)
                for j in 0..self.config.hidden_size
                {
                    for i in 0..self.config.input_features
                    {
                        self.w2[j][i] += lr * error[i] * hidden[j];
                    }
                }
                // Update b2
                for i in 0..self.config.input_features
                {
                    self.b2[i] += lr * error[i];
                }

                // Backprop: update W1 (encoder)
                for i in 0..self.config.input_features
                {
                    for j in 0..self.config.hidden_size
                    {
                        let mut delta = 0.0;
                        for k in 0..self.config.input_features
                        {
                            delta += error[k] * self.w2[j][k];
                        }
                        // ReLU derivative
                        let relu_grad = if hidden[j] > 0.0 { 1.0 } else { 0.0 };
                        let grad = delta * relu_grad * sample[i];
                        self.w1[i][j] += lr * grad;
                    }
                }
                // Update b1
                for j in 0..self.config.hidden_size
                {
                    let mut delta = 0.0;
                    for k in 0..self.config.input_features
                    {
                        delta += error[k] * self.w2[j][k];
                    }
                    let relu_grad = if hidden[j] > 0.0 { 1.0 } else { 0.0 };
                    self.b1[j] += lr * delta * relu_grad;
                }

                self.sample_count += 1;
            }

            self.training_mse = epoch_mse / normal_data.len() as f64;
        }

        // Calibrer le seuil: mean + 2*std des erreurs d'entraînement
        let errors: Vec<f64> = normal_data
            .iter()
            .map(|sample| {
                let (_, output) = self.forward(sample);
                self.reconstruction_mse(sample, &output)
            })
            .collect();

        let mean: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let variance: f64 =
            errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64;
        self.calibrated_threshold = mean + 3.0 * variance.sqrt();
    }

    /// Score d'anomalie pour un échantillon.
    pub fn anomaly_score(&self, input: &[f64]) -> f64 {
        let (_, output) = self.forward(input);
        self.reconstruction_mse(input, &output)
    }

    /// Vérifier si un échantillon est anormal.
    pub fn is_anomaly(&self, input: &[f64]) -> bool {
        self.anomaly_score(input) > self.calibrated_threshold
    }

    /// Nombre d'échantillons d'entraînement.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// MSE d'entraînement.
    pub fn training_mse(&self) -> f64 {
        self.training_mse
    }

    /// Seuil calibré.
    pub fn threshold(&self) -> f64 {
        self.calibrated_threshold
    }
}

/// Moteur d'apprentissage pour l'IDS.
///
/// Entraîne le modèle d'anomalie sur du trafic labellisé comme normal,
/// puis score les nouveaux flux pour détecter les anomalies.
pub struct IdsLearner {
    pub config: LearnerConfig,
    model: AnomalyModel,
    /// Buffer d'entraînement (flux normaux)
    normal_buffer: Vec<Vec<f64>>,
    /// Buffer d'anomalies (pour évaluation)
    anomaly_buffer: Vec<Vec<f64>>,
    /// État d'entraînement
    is_trained: bool,
    /// Scores d'anomalie récents
    recent_scores: Vec<f64>,
}

impl IdsLearner {
    pub fn new(config: LearnerConfig) -> Self {
        Self {
            model: AnomalyModel::new(config.clone()),
            normal_buffer: Vec::new(),
            anomaly_buffer: Vec::new(),
            is_trained: false,
            recent_scores: Vec::new(),
            config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(LearnerConfig::default())
    }

    /// Ajouter des échantillons normaux au buffer d'entraînement.
    pub fn add_normal_samples(&mut self, samples: Vec<Vec<f64>>) {
        self.normal_buffer.extend(samples);
    }

    /// Ajouter des échantillons d'anomalie (pour évaluation).
    pub fn add_anomaly_samples(&mut self, samples: Vec<Vec<f64>>) {
        self.anomaly_buffer.extend(samples);
    }

    /// Entraîner le modèle sur les données accumulées.
    pub fn train(&mut self) {
        if self.normal_buffer.len() < 10
        {
            return;
        }

        // Normaliser les données (min-max sur chaque feature)
        let n_features = self.config.input_features;
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];

        for sample in &self.normal_buffer
        {
            for (i, &v) in sample.iter().enumerate().take(n_features)
            {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }

        let normalized: Vec<Vec<f64>> = self
            .normal_buffer
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let range = maxs[i] - mins[i];
                        if range < f64::EPSILON
                        {
                            0.0
                        }
                        else
                        {
                            (v - mins[i]) / range
                        }
                    })
                    .collect()
            })
            .collect();

        self.model.train(&normalized);
        self.is_trained = true;
    }

    /// Classifier un flux: score d'anomalie.
    pub fn score(&mut self, features: &[f64]) -> f64 {
        if !self.is_trained
        {
            return 0.0;
        }
        let score = self.model.anomaly_score(features);
        self.recent_scores.push(score);
        if self.recent_scores.len() > 1000
        {
            self.recent_scores.remove(0);
        }
        score
    }

    /// Vérifier si un flux est anormal.
    pub fn detect(&mut self, features: &[f64]) -> Option<DetectorResult> {
        if !self.is_trained
        {
            return None;
        }

        let score = self.score(features);
        if score > self.model.calibrated_threshold
        {
            let confidence = ((score - self.model.calibrated_threshold)
                / self.model.calibrated_threshold)
                .min(1.0) as f32;
            let severity = if confidence >= 0.9
            {
                "CRITICAL"
            }
            else if confidence >= 0.7
            {
                "WARNING"
            }
            else
            {
                "INFO"
            };

            Some(DetectorResult {
                detector: "ml_anomaly".to_string(),
                label_en: "ml_detected_anomaly".to_string(),
                label_fr: "anomalie_ml".to_string(),
                confidence,
                severity: severity.to_string(),
                source_ip: String::new(),
                destination_ip: String::new(),
                details: format!(
                    "anomaly_score={:.4} threshold={:.4}",
                    score, self.model.calibrated_threshold
                ),
            })
        }
        else
        {
            None
        }
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn model(&self) -> &AnomalyModel {
        &self.model
    }

    pub fn normal_buffer_len(&self) -> usize {
        self.normal_buffer.len()
    }

    pub fn recent_scores(&self) -> &[f64] {
        &self.recent_scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_samples(n: usize, features: usize) -> Vec<Vec<f64>> {
        let mut samples = Vec::new();
        for i in 0..n
        {
            let sample: Vec<f64> = (0..features)
                .map(|j| ((i + j) as f64 * 0.1).sin())
                .collect();
            samples.push(sample);
        }
        samples
    }

    #[test]
    fn test_anomaly_model_train_and_score() {
        let config = LearnerConfig {
            input_features: 4,
            hidden_size: 8,
            epochs: 100,
            ..Default::default()
        };
        let mut model = AnomalyModel::new(config);

        let normal = make_normal_samples(100, 4);
        model.train(&normal);

        // Normal sample should have low score
        let normal_score = model.anomaly_score(&normal[0]);
        assert!(
            normal_score < model.calibrated_threshold,
            "normal score should be below threshold"
        );

        // Anomalous sample should have high score
        let anomaly = vec![100.0, 100.0, 100.0, 100.0];
        let anomaly_score = model.anomaly_score(&anomaly);
        assert!(
            anomaly_score > normal_score,
            "anomaly score should be higher"
        );
    }

    #[test]
    fn test_ids_learner_detect() {
        let config = LearnerConfig {
            input_features: 4,
            hidden_size: 8,
            epochs: 100,
            ..Default::default()
        };
        let mut learner = IdsLearner::new(config);

        let normal = make_normal_samples(100, 4);
        learner.add_normal_samples(normal);
        learner.train();

        assert!(learner.is_trained());

        // Normal sample: no detection (within training distribution)
        let normal_sample: Vec<f64> = (0..4).map(|j| ((10 + j) as f64 * 0.1).sin()).collect();
        assert!(learner.detect(&normal_sample).is_none());

        // Anomalous sample: detection (far outside training distribution)
        let anomaly_sample = vec![100.0; 4];
        let result = learner.detect(&anomaly_sample);
        assert!(result.is_some());
        assert_eq!(result.unwrap().detector, "ml_anomaly");
    }

    #[test]
    fn test_ids_learner_untrained_returns_none() {
        let mut learner = IdsLearner::with_defaults();
        let result = learner.detect(&[0.0; 10]);
        assert!(result.is_none());
    }
}
