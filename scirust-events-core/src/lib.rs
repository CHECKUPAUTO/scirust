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
        if self.current_offset + self.window_size > self.data.len() {
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
