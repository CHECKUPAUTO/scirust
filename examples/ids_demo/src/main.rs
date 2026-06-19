use rand::Rng;
use scirust_ids::*;

fn main() {
    println!("=== SciRust IDS Demo ===\n");

    let mut engine = IdsEngine::with_defaults();
    let mut alert_log = AlertLog::with_defaults();

    // --- Scénario 1: Scan de ports ---
    println!("--- Scénario 1: Scan de ports vertical ---");
    let mut window = FlowWindow::new(0.0, 60.0);
    for port in 1..=40
    {
        let mut f = Flow::new("192.168.1.100", "10.0.0.5", 40000 + port, port);
        f.start_time = port as f64 * 0.5;
        f.end_time = f.start_time + 0.05;
        f.packets_out = 1;
        f.bytes_out = 60;
        f.bytes_in = 0;
        f.packets_in = 0;
        window.push(f);
    }

    let report = engine.analyze(&window, 1000.0);
    print_report(&report);
    alert_log.push_results(&report.results, report.timestamp);

    // --- Scénario 2: SYN flood ---
    println!("\n--- Scénario 2: SYN flood (DDoS) ---");
    let mut window2 = FlowWindow::new(0.0, 10.0);
    let mut rng = rand::thread_rng();
    for src in 0..25
    {
        for _ in 0..15
        {
            let octet = rng.gen_range(1..254);
            let mut f = Flow::new(
                &format!("10.0.{}.{}", src, octet),
                "10.0.0.1",
                rng.gen_range(40000..60000),
                80,
            );
            f.syn_count = 3;
            f.packets_out = 4;
            f.packets_in = 0;
            f.bytes_out = 240;
            f.bytes_in = 0;
            f.start_time = rng.gen_range(0.0..10.0);
            f.end_time = f.start_time + 0.1;
            window2.push(f);
        }
    }

    let report2 = engine.analyze(&window2, 2000.0);
    print_report(&report2);
    alert_log.push_results(&report2.results, report2.timestamp);

    // --- Scénario 3: Brute force SSH ---
    println!("\n--- Scénario 3: Brute force SSH ---");
    let mut window3 = FlowWindow::new(0.0, 60.0);
    for i in 0..12
    {
        let mut f = Flow::new("203.0.113.50", "10.0.0.10", 50000 + i, 22);
        f.syn_count = 1;
        f.rst_count = 1;
        f.packets_out = 2;
        f.packets_in = 1;
        f.bytes_out = 120;
        f.bytes_in = 60;
        f.start_time = i as f64 * 4.0;
        f.end_time = f.start_time + 0.3;
        window3.push(f);
    }

    let report3 = engine.analyze(&window3, 3000.0);
    print_report(&report3);
    alert_log.push_results(&report3.results, report3.timestamp);

    // --- Scénario 4: Beaconing C2 ---
    println!("\n--- Scénario 4: Beaconing C2 ---");
    let mut window4 = FlowWindow::new(0.0, 600.0);
    for i in 0..15
    {
        let mut f = Flow::new("10.0.0.20", "45.33.32.156", 60000 + i, 443);
        f.bytes_out = 256;
        f.bytes_in = 512;
        f.packets_out = 2;
        f.packets_in = 2;
        f.start_time = i as f64 * 40.0;
        f.end_time = f.start_time + 0.5;
        window4.push(f);
    }

    let report4 = engine.analyze(&window4, 4000.0);
    print_report(&report4);
    alert_log.push_results(&report4.results, report4.timestamp);

    // --- Résumé ---
    println!("\n=== Résumé ===");
    println!("Total événements: {}", engine.total_events());
    println!("Total alertes log: {}", alert_log.len());
    let crits = alert_log.by_severity(AlertSeverity::Critical);
    println!("Alertes critiques: {}", crits.len());

    if !alert_log.is_empty()
    {
        println!("\nDernières alertes:");
        for alert in alert_log.recent(5)
        {
            println!(
                "  [{}] {} ({}) <- {} | {}",
                alert.severity.label_en(),
                alert.attack_type,
                alert.attack_type_fr,
                alert.source_ip,
                alert.details
            );
        }
    }
}

fn print_report(report: &IdsReport) {
    println!(
        "  Flux analysés: {} | Alertes: {} | Critiques: {}",
        report.flows_analyzed, report.alert_count, report.critical_count
    );
    for r in &report.results
    {
        println!(
            "    [{}] {} ({}) conf={:.2} src={} | {}",
            r.severity, r.label_en, r.label_fr, r.confidence, r.source_ip, r.details
        );
    }
}
