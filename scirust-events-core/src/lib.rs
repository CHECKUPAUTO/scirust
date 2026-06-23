use scirust_core::autodiff::reverse::Tensor;
use serde::{Deserialize, Serialize};

/// Représente un événement détecté avec ses métadonnées.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: u64,
    pub timestamp: f64,
    pub label_en: String,
    pub label_fr: String,
    pub confidence: f32,
    pub data_snapshot: Option<Vec<f32>>,
}

/// Flux de données temporelles segmenté en fenêtres déterministes.
pub struct EventStream {
    pub data: Vec<f32>,
    pub window_size: usize,
    pub stride: usize,
    pub current_offset: usize,
}

impl EventStream {
    pub fn new(data: Vec<f32>, window_size: usize, stride: usize) -> Self {
        Self {
            data,
            window_size,
            stride,
            current_offset: 0,
        }
    }

    pub fn next_window(&mut self) -> Option<Tensor> {
        if self.current_offset + self.window_size > self.data.len()
        {
            return None;
        }
        let chunk = &self.data[self.current_offset..self.current_offset + self.window_size];
        let t = Tensor::from_vec(chunk.to_vec(), 1, self.window_size);
        self.current_offset += self.stride;
        Some(t)
    }
}

/// Trait pour tout détecteur d'anomalies ou d'événements.
pub trait EventDetector {
    /// Retourne un score de détection (0.0 à 1.0) et une étiquette.
    fn detect(&mut self, window: &Tensor) -> (f32, String, String);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_stream_windows_deterministically() {
        let mut s = EventStream::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(s.next_window().unwrap().data, vec![1.0, 2.0, 3.0]);
        assert_eq!(s.next_window().unwrap().data, vec![3.0, 4.0, 5.0]);
        assert!(s.next_window().is_none());
    }

    struct SumThreshold(f32);
    impl EventDetector for SumThreshold {
        fn detect(&mut self, window: &Tensor) -> (f32, String, String) {
            let sum: f32 = window.data.iter().sum();
            let score = if sum >= self.0 { 1.0 } else { 0.0 };
            (score, "spike".to_string(), "pic".to_string())
        }
    }

    #[test]
    fn detector_trait_scores_a_window() {
        let mut d = SumThreshold(10.0);
        let w = Tensor::from_vec(vec![4.0, 4.0, 4.0], 1, 3);
        let (score, en, fr) = d.detect(&w);
        assert_eq!(score, 1.0);
        assert_eq!((en.as_str(), fr.as_str()), ("spike", "pic"));
    }
}
