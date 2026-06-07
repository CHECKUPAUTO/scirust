use scirust_core::autodiff::reverse::Tensor;
use scirust_events_core::{Event, EventDetector, EventStream};
use scirust_runtime::{save_weights, load_weights};
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

        while let Some(window) = stream.next_window() {
            let (score, en, fr) = self.detector.detect(&window);
            if score >= 1.0 {
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
    pub fn save_detector_state(&self, path: &str, params: &HashMap<String, Tensor>) -> io::Result<()> {
        save_weights(params, path)
    }

    /// Charge l'état du détecteur depuis un fichier SRT1.
    pub fn load_detector_state(&self, path: &str) -> io::Result<HashMap<String, Tensor>> {
        load_weights(path)
    }
}
