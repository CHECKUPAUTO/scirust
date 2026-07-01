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

        // La fenêtre est non vide (garde ci-dessus) : on initialise le max avec
        // le premier élément pour que les fenêtres entièrement négatives soient
        // correctement prises en compte (ne pas biaiser vers 0.0).
        let mut max_val = window.data[0];
        for &x in &window.data[1..]
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

        // Sortie vide : rien à classer, on renvoie une confiance nulle.
        if output.data.is_empty()
        {
            return (0.0, "unknown".into(), "inconnu".into());
        }

        // Recherche de l'index du max (argmax)
        let mut max_idx = 0;
        let mut max_val = output.data[0];
        for (i, &val) in output.data.iter().enumerate()
        {
            if val > max_val
            {
                max_val = val;
                max_idx = i;
            }
        }

        // Le contrat EventDetector impose un score dans 0.0..=1.0. Les sorties du
        // modèle sont des logits bruts ; on les convertit en probabilité via un
        // softmax numériquement stable (décalé par le max) et on renvoie la
        // probabilité de la classe argmax, garantie dans (0.0, 1.0].
        let mut sum_exp = 0.0f32;
        for &val in &output.data
        {
            sum_exp += (val - max_val).exp();
        }
        // sum_exp >= 1.0 (le terme argmax vaut exp(0) = 1), donc pas de division
        // par zéro et le résultat reste borné par 1.0.
        let confidence = 1.0 / sum_exp;

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

        (confidence, en, fr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::Tensor;
    use scirust_core::nn::{Linear, PcgEngine, SmallNormal, Zeros};

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
        // All-zero logits → uniform softmax over 4 classes → 1/4 confidence
        // (the score is a probability in (0.0, 1.0], not a raw logit).
        assert!((score - 0.25).abs() < 1e-6);
        assert_eq!((en.as_str(), fr.as_str()), ("a", "A"));
    }

    #[test]
    fn spike_detector_handles_all_negative_window() {
        // Regression: the max used to be seeded at 0.0, so an all-negative window
        // reported 0.0 as its peak instead of the true (negative) maximum, which
        // then leaked into the EMA. With alpha=1.0 the EMA equals the window max,
        // so we can pin it exactly: max(-5, -3, -8) = -3, not 0.
        let mut d = SpikeDetector::new(1.0, 1.0);
        let (score, en, _) = d.detect(&Tensor::from_vec(vec![-5.0, -3.0, -8.0], 1, 3));
        assert!((d.ema - (-3.0)).abs() < 1e-6);
        assert_eq!(en, "background");
        // score = ema / threshold = -3 / 1 = -3 (negative peak, not a fake 0).
        assert!((score - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn classifier_confidence_stays_within_the_unit_range() {
        // Regression: the classifier used to return the raw argmax logit as the
        // confidence, violating the EventDetector 0.0..=1.0 contract. Even with a
        // biased model whose logits are far outside that range, the reported score
        // must now be a softmax probability inside (0.0, 1.0].
        let mut rng = PcgEngine::new(7);
        // Zeros weights + a large constant... but only Zeros bias is available, so
        // use a deterministic SmallNormal weight to get non-degenerate logits and
        // assert the *property* (bounded probability), independent of the argmax.
        let model =
            Sequential::new().add(Linear::new(3, 5, &SmallNormal::new(3.0), &Zeros, &mut rng));
        let mut classifier = EventClassifier::new(
            model,
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
        );
        let (score, _, _) = classifier.detect(&Tensor::from_vec(vec![5.0, -4.0, 6.0], 1, 3));
        assert!(score > 0.0 && score <= 1.0, "score {score} not in (0,1]");
    }
}
