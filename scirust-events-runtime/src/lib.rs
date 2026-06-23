use scirust_core::autodiff::reverse::Tensor;
use scirust_events_core::{Event, EventDetector, EventStream};
use scirust_runtime::{load_weights, save_weights};
use std::collections::HashMap;
use std::io;

/// Gère l'exécution déterministe de la détection d'événements.
pub struct EventRuntime {
    pub detector: Box<dyn EventDetector>,
}

impl EventRuntime {
    pub fn new(detector: Box<dyn EventDetector>) -> Self {
        Self { detector }
    }

    /// Exécute la détection sur un flux complet et retourne les événements trouvés.
    pub fn process_all(&mut self, stream: &mut EventStream) -> Vec<Event> {
        let mut events = Vec::new();
        let mut count = 0;

        while let Some(window) = stream.next_window()
        {
            let (score, en, fr) = self.detector.detect(&window);
            if score >= 1.0
            {
                events.push(Event {
                    id: count,
                    timestamp: stream.current_offset as f64,
                    label_en: en,
                    label_fr: fr,
                    confidence: score,
                    data_snapshot: Some(window.data.clone()),
                });
                count += 1;
            }
        }
        events
    }

    /// Sauvegarde l'état du détecteur au format SRT1.
    pub fn save_detector_state(
        &self,
        path: &str,
        params: &HashMap<String, Tensor>,
    ) -> io::Result<()> {
        save_weights(params, path)
    }

    /// Charge l'état du détecteur depuis un fichier SRT1.
    pub fn load_detector_state(&self, path: &str) -> io::Result<HashMap<String, Tensor>> {
        load_weights(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumThreshold(f32);
    impl EventDetector for SumThreshold {
        fn detect(&mut self, window: &Tensor) -> (f32, String, String) {
            let sum: f32 = window.data.iter().sum();
            let score = if sum >= self.0 { 1.0 } else { 0.0 };
            (score, "spike".to_string(), "pic".to_string())
        }
    }

    #[test]
    fn runtime_detects_events_above_threshold() {
        let mut rt = EventRuntime::new(Box::new(SumThreshold(10.0)));
        let mut stream = EventStream::new(vec![1.0, 1.0, 6.0, 6.0, 0.0, 0.0], 2, 2);
        let events = rt.process_all(&mut stream);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].label_en, "spike");
        assert!(events[0].confidence >= 1.0);
    }

    #[test]
    fn detector_state_round_trips_through_srt1() {
        let rt = EventRuntime::new(Box::new(SumThreshold(1.0)));
        let mut params = HashMap::new();
        params.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let path = std::env::temp_dir()
            .join("scirust_events_rt_test.srt")
            .to_string_lossy()
            .to_string();
        rt.save_detector_state(&path, &params).unwrap();
        let loaded = rt.load_detector_state(&path).unwrap();
        assert_eq!(loaded["w"].data, vec![1.0, 2.0, 3.0]);
        let _ = std::fs::remove_file(&path);
    }
}
