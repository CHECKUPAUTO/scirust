use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Module, Sequential};
use scirust_events_core::EventDetector;

/// Détecteur de spikes utilisant un lissage exponentiel (EMA).
pub struct SpikeDetector {
    pub threshold: f32,
    pub alpha: f32,
    pub ema: f32,
}

impl SpikeDetector {
    pub fn new(threshold: f32, alpha: f32) -> Self {
        Self {
            threshold,
            alpha,
            ema: 0.0,
        }
    }
}

impl EventDetector for SpikeDetector {
    fn detect(&mut self, window: &Tensor) -> (f32, String, String) {
        if window.data.is_empty()
        {
            return (0.0, "none".into(), "aucun".into());
        }

        let mut max_val = 0.0f32;
        for &x in &window.data
        {
            if x > max_val
            {
                max_val = x;
            }
        }

        // Mise à jour de l'EMA pour un seuillage adaptatif ou lissage
        self.ema = self.alpha * max_val + (1.0 - self.alpha) * self.ema;

        if self.ema > self.threshold
        {
            (1.0, "spike".into(), "pic".into())
        }
        else
        {
            (
                self.ema / self.threshold,
                "background".into(),
                "bruit".into(),
            )
        }
    }
}

/// Classifieur d'événements utilisant un modèle SciRust Sequential.
pub struct EventClassifier {
    pub model: Sequential,
    pub labels_en: Vec<String>,
    pub labels_fr: Vec<String>,
}

impl EventClassifier {
    pub fn new(model: Sequential, labels_en: Vec<String>, labels_fr: Vec<String>) -> Self {
        Self {
            model,
            labels_en,
            labels_fr,
        }
    }
}

impl EventDetector for EventClassifier {
    fn detect(&mut self, window: &Tensor) -> (f32, String, String) {
        let tape = Tape::new();
        let input = tape.input(window.clone());
        let output_var = self.model.forward(&tape, input);
        let output = output_var.tape().value(output_var.idx());

        // Recherche de l'index du max (argmax)
        let mut max_idx = 0;
        let mut max_val = -f32::INFINITY;
        for (i, &val) in output.data.iter().enumerate()
        {
            if val > max_val
            {
                max_val = val;
                max_idx = i;
            }
        }

        let en = self
            .labels_en
            .get(max_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        let fr = self
            .labels_fr
            .get(max_idx)
            .cloned()
            .unwrap_or_else(|| "inconnu".into());

        (max_val, en, fr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::Tensor;
    use scirust_core::nn::{KaimingNormal, Linear, PcgEngine, ReLU, Zeros};

    #[test]
    fn test_spike_detector_ema() {
        let mut detector = SpikeDetector::new(1.0, 0.5);
        let data = vec![2.0, 2.0];
        let tensor = Tensor::from_vec(data, 1, 2);

        // Premier appel: ema = 0.5*2 + 0.5*0 = 1.0
        let (score, _, _) = detector.detect(&tensor);
        assert!(score >= 1.0);
        assert_eq!(detector.ema, 1.0);

        let data2 = vec![0.0, 0.0];
        let tensor2 = Tensor::from_vec(data2, 1, 2);
        // Second appel: ema = 0.5*0 + 0.5*1 = 0.5
        let (score2, _, _) = detector.detect(&tensor2);
        assert_eq!(score2, 0.5);
    }

    #[test]
    fn test_event_classifier() {
        let mut rng = PcgEngine::new(42);
        let mut model = Sequential::new()
            .add(Linear::new(2, 2, &KaimingNormal, &Zeros, &mut rng))
            .add(ReLU::new());

        let mut classifier = EventClassifier::new(
            model,
            vec!["A".into(), "B".into()],
            vec!["Alpha".into(), "Beta".into()],
        );

        let data = vec![1.0, 0.5];
        let tensor = Tensor::from_vec(data, 1, 2);
        let (score, en, fr) = classifier.detect(&tensor);
        assert!(score >= 0.0);
        assert!(!en.is_empty());
        assert!(!fr.is_empty());
    }
}
