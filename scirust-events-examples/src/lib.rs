use scirust_events_core::EventStream;
use scirust_events_models::SpikeDetector;
use scirust_events_runtime::EventRuntime;

pub fn run_example() {
    let data = vec![0.1, 0.2, 1.5, 0.3, 0.1, 2.0, 0.2];
    let mut stream = EventStream::new(data, 2, 1);
    let detector = SpikeDetector::new(1.0, 0.8);
    let mut runtime = EventRuntime::new(Box::new(detector));

    let events = runtime.process_all(&mut stream);
    for e in events
    {
        println!(
            "Event (EN: {}, FR: {}) at t={} confidence={}",
            e.label_en, e.label_fr, e.timestamp, e.confidence
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_example_logic() {
        run_example();
    }
}
