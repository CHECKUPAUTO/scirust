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
    use scirust_core::nn::{Linear, PcgEngine, Zeros};

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
    fn spike_detector_flags_a_threshold_crossing() {
        let mut d = SpikeDetector::new(5.0, 0.5);
        // max 4 → ema = 0.5·4 = 2.0 ≤ 5 → background, score 2/5 = 0.4.
        let (s1, en1, _) = d.detect(&Tensor::from_vec(vec![1.0, 4.0, 2.0], 1, 3));
        assert_eq!(en1, "background");
        assert!((s1 - 0.4).abs() < 1e-6);
        // max 12 → ema = 0.5·12 + 0.5·2 = 7.0 > 5 → spike, score 1.0.
        let (s2, en2, fr2) = d.detect(&Tensor::from_vec(vec![12.0], 1, 1));
        assert_eq!((en2.as_str(), fr2.as_str()), ("spike", "pic"));
        assert_eq!(s2, 1.0);
        assert!((d.ema - 7.0).abs() < 1e-6);
    }

    #[test]
    fn classifier_argmax_maps_to_the_right_label() {
        let mut rng = PcgEngine::new(1);
        // Zeros weight + bias → the model outputs all zeros, so argmax picks the
        // first index → label 0. This pins the argmax→label mapping exactly
        // (the previous test only checked the labels were non-empty).
        let model = Sequential::new().add(Linear::new(3, 4, &Zeros, &Zeros, &mut rng));
        let mut classifier = EventClassifier::new(
            model,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
        );
        let (score, en, fr) = classifier.detect(&Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        assert_eq!(score, 0.0);
        assert_eq!((en.as_str(), fr.as_str()), ("a", "A"));
    }
}
