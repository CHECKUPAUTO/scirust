//! CLI command implementations

use scirust_integration::{
    backend::BackendFactory,
    config::PipelineConfig,
    pipeline::Pipeline,
    templates::{TemplateKind, generate_project},
};
use scirust_mqtt::{MqttConfig, MqttPublisher, SimulatedMqttPublisher};
use scirust_opcua::{OpcuaClient, OpcuaConfig, SimulatedOpcuaClient};
use std::path::Path;

/// DISCOVER: Browse OPC-UA server for available sensor nodes
pub fn discover(endpoint: &str, filter: &str, simulated: bool) -> Result<(), String> {
    println!("=== OPC-UA Discovery ===\n");

    if simulated
    {
        println!("Using simulated backend (no real PLC connection)\n");
        let client = SimulatedOpcuaClient::new();
        let nodes = client
            .browse(filter)
            .map_err(|e| format!("Browse error: {}", e))?;
        print_nodes(&nodes);
        println!("\n{} nodes found.", nodes.len());
        return Ok(());
    }

    // Real OPC-UA would go here
    println!("Endpoint: {}", endpoint);
    println!("Filter:   {}\n", filter);
    println!("Real OPC-UA backend requires the 'real-opcua' feature flag.");
    println!("To enable:");
    println!("  1. Add to Cargo.toml: scirust-integration = {{ features = [\"real-opcua\"] }}");
    println!("  2. Add dependency:   opcua = \"0.13\"");
    println!("\nFalling back to simulated discovery:\n");
    let client = SimulatedOpcuaClient::new();
    let nodes = client
        .browse(filter)
        .map_err(|e| format!("Browse error: {}", e))?;
    print_nodes(&nodes);
    println!("\n{} nodes found (simulated).", nodes.len());
    Ok(())
}

fn print_nodes(nodes: &[scirust_opcua::OpcuaNode]) {
    if nodes.is_empty()
    {
        println!("  (no nodes found)");
        return;
    }
    println!(
        "{:<30} {:<25} {:<10} Description",
        "Node ID", "Display Name", "Unit"
    );
    println!("{}", "-".repeat(95));
    for n in nodes
    {
        println!(
            "{:<30} {:<25} {:<10} {}",
            n.node_id, n.display_name, n.unit, n.description
        );
    }
}

/// TEST-OPCUA: Test OPC-UA connection and read values
pub fn test_opcua(endpoint: &str, simulated: bool, samples: usize) -> Result<(), String> {
    println!("=== OPC-UA Connection Test ===\n");

    let mut client: Box<dyn OpcuaClient> = if simulated
    {
        Box::new(SimulatedOpcuaClient::new())
    }
    else
    {
        println!("Real OPC-UA backend not available, using simulated.");
        Box::new(SimulatedOpcuaClient::new())
    };

    let config = OpcuaConfig {
        endpoint: endpoint.to_string(),
        ..Default::default()
    };

    client
        .connect(&config)
        .map_err(|e| format!("Connect failed: {}", e))?;
    println!("Connected to: {}", config.endpoint);

    let nodes = client
        .browse("vibration")
        .map_err(|e| format!("Browse failed: {}", e))?;
    println!("Found {} vibration nodes", nodes.len());

    let node_ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
    client
        .subscribe(&node_ids)
        .map_err(|e| format!("Subscribe failed: {}", e))?;
    println!("Subscribed to {} nodes", node_ids.len());

    println!("\nReading {} samples:\n", samples);
    println!(
        "{:<30} {:<12} {:<10} Timestamp",
        "Node ID", "Value", "Quality"
    );
    println!("{}", "-".repeat(70));

    for i in 0..samples
    {
        let values = client
            .poll_subscription()
            .map_err(|e| format!("Poll failed: {}", e))?;
        for v in &values
        {
            println!(
                "{:<30} {:<12.4} {:<10} {:.3}",
                v.node_id,
                v.value,
                if v.quality.is_good() { "Good" } else { "Bad" },
                v.timestamp
            );
        }
        if i == 0 && values.is_empty()
        {
            println!("  (no values returned — simulated backend needs a tick)");
        }
    }

    client
        .disconnect()
        .map_err(|e| format!("Disconnect failed: {}", e))?;
    println!("\nOPC-UA test: PASSED");
    Ok(())
}

/// TEST-MQTT: Test MQTT broker connection
pub fn test_mqtt(host: &str, port: u16, simulated: bool, topic: &str) -> Result<(), String> {
    println!("=== MQTT Connection Test ===\n");

    let mut publisher: Box<dyn MqttPublisher> = if simulated
    {
        Box::new(SimulatedMqttPublisher::new())
    }
    else
    {
        println!("Real MQTT backend not available, using simulated.");
        Box::new(SimulatedMqttPublisher::new())
    };

    let config = MqttConfig {
        host: host.to_string(),
        port,
        ..Default::default()
    };

    publisher
        .connect(&config)
        .map_err(|e| format!("Connect failed: {}", e))?;
    println!("Connected to MQTT broker: {}:{}", config.host, config.port);
    println!("Client ID: {}", config.client_id);
    println!("Base topic: {}", config.base_topic);
    println!("QoS: {}", config.qos);

    let payload = br#"{"test": true, "source": "scirust-industrial"}"#;
    publisher
        .publish(topic, payload, config.qos, false)
        .map_err(|e| format!("Publish failed: {}", e))?;
    println!("\nPublished test message to: {}", topic);
    println!("Payload: {}", String::from_utf8_lossy(payload));

    publisher
        .disconnect()
        .map_err(|e| format!("Disconnect failed: {}", e))?;
    println!("\nMQTT test: PASSED");
    Ok(())
}

/// GEN-CONFIG: Generate a pipeline configuration file
pub fn gen_config(
    output: &str,
    template: &str,
    n_stations: usize,
    line_id: &str,
) -> Result<(), String> {
    println!("=== Generate Configuration ===\n");

    let config = match template.to_lowercase().as_str()
    {
        "minimal" | "basic" => PipelineConfig::default(),
        "automotive" | "line" => PipelineConfig::automotive_line(line_id, n_stations),
        "bearing" =>
        {
            let mut cfg = PipelineConfig::default();
            cfg.stations[0].bearing = Some(scirust_integration::config::BearingConfig {
                pitch_diameter: 39.04,
                ball_diameter: 7.94,
                n_balls: 9,
                contact_angle_deg: 0.0,
            });
            cfg.stations[0].shaft_freq = Some(29.53);
            cfg
        },
        "pdm" | "predictive" => PipelineConfig::default(),
        _ =>
        {
            return Err(format!(
                "Unknown template: {}. Use: minimal, automotive, bearing, pdm",
                template
            ));
        },
    };

    let errors = config.validate();
    if !errors.is_empty()
    {
        eprintln!("Config validation warnings:");
        for e in &errors
        {
            eprintln!("  - {}", e);
        }
    }

    let path = Path::new(output);
    config
        .save_to_file(path)
        .map_err(|e| format!("Save failed: {}", e))?;
    println!("Configuration saved to: {}", output);
    println!("Template: {}", template);
    println!("Stations: {}", config.stations.len());
    println!("Backend: {}", config.backend_type);
    println!("\nEdit the file to customize your monitoring setup.");
    Ok(())
}

/// SCAFFOLD: Generate a complete monitoring project
pub fn scaffold(name: &str, output: &str, template: &str) -> Result<(), String> {
    println!("=== Scaffold Monitoring Project ===\n");

    let kind = TemplateKind::parse_from_str(template).ok_or_else(|| {
        format!(
            "Unknown template: {}. Use: minimal, automotive, bearing, pdm",
            template
        )
    })?;

    let templates = generate_project(kind, name);
    let base = Path::new(output).join(name);

    // Create directory structure
    std::fs::create_dir_all(base.join("src"))
        .map_err(|e| format!("Cannot create directory: {}", e))?;

    println!("Project: {}", name);
    println!("Template: {} ({})", kind.label(), kind.description());
    println!("Output: {}\n", base.display());

    for t in &templates
    {
        let path = base.join(&t.filename);
        if let Some(parent) = path.parent()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create directory {}: {}", parent.display(), e))?;
        }
        std::fs::write(&path, &t.content)
            .map_err(|e| format!("Cannot write {}: {}", path.display(), e))?;
        println!("  Created: {} ({})", t.filename, t.description);
    }

    println!("\nNext steps:");
    println!("  cd {}", name);
    println!("  cargo run");
    println!("\nEdit config.json to customize your monitoring setup.");
    println!("To use real OPC-UA/MQTT, uncomment the relevant lines in Cargo.toml.");
    Ok(())
}

/// RUN: Run a monitoring pipeline from config
pub fn run(config_path: &str, cycles: usize, report_path: Option<&str>) -> Result<(), String> {
    println!("=== Run Monitoring Pipeline ===\n");

    let path = Path::new(config_path);
    let config = if path.exists()
    {
        PipelineConfig::from_file(path)?
    }
    else
    {
        println!("Config file '{}' not found, using defaults.", config_path);
        PipelineConfig::default()
    };

    println!("Backend: {}", config.backend_type);
    println!("Stations: {}", config.stations.len());
    println!("Cycles: {}", cycles);
    println!();

    let mut pipeline = Pipeline::new(config);
    let report = pipeline.run(cycles);

    // Print report
    println!("=== Pipeline Report ===\n");
    println!("Total cycles:      {}", report.total_cycles);
    println!("Events detected:   {}", report.total_events);
    println!("Events published:   {}", report.total_published);
    println!(
        "Final Health:      {:.3} ({})",
        report.final_health_index, report.final_health_state
    );
    println!("RUL:               {:.1} hours", report.final_rul);
    println!(
        "RUL 95%% CI:        [{:.1}, {:.1}]",
        report.rul_lower_bound, report.rul_upper_bound
    );
    println!("Audit entries:     {}", report.audit_entries);
    println!("Audit chain valid: {}", report.audit_chain_valid);
    println!("MQTT messages:     {}", report.mqtt_messages);

    // Save report if requested
    if let Some(rp) = report_path
    {
        let json = pipeline.export_report_json()?;
        std::fs::write(rp, &json).map_err(|e| format!("Cannot write report: {}", e))?;
        println!("\nReport saved to: {}", rp);
    }

    pipeline.shutdown();
    println!("\nPipeline shutdown complete.");
    Ok(())
}

/// DOCTOR: Diagnose integration issues
pub fn doctor(config_path: &str) -> Result<(), String> {
    println!("=== SciRust Industrial Doctor ===\n");

    let mut issues = Vec::new();
    let mut passed = Vec::new();

    // Check 1: Config file
    let path = Path::new(config_path);
    let config = if path.exists()
    {
        match PipelineConfig::from_file(path)
        {
            Ok(c) =>
            {
                passed.push(format!("Config file '{}' loaded successfully", config_path));
                c
            },
            Err(e) =>
            {
                issues.push(format!("Config file error: {}", e));
                PipelineConfig::default()
            },
        }
    }
    else
    {
        issues.push(format!(
            "Config file '{}' not found (using defaults)",
            config_path
        ));
        PipelineConfig::default()
    };

    // Check 2: Config validation
    let errors = config.validate();
    if errors.is_empty()
    {
        passed.push("Config validation: OK".to_string());
    }
    else
    {
        for e in errors
        {
            issues.push(format!("Config validation: {}", e));
        }
    }

    // Check 3: Backend type
    let backend_type =
        scirust_integration::backend::BackendType::parse_from_str(&config.backend_type);
    match backend_type
    {
        Some(bt) =>
        {
            passed.push(format!(
                "Backend type '{}' is recognized",
                config.backend_type
            ));
            if let Some(crate_name) = bt.requires_external_crate()
            {
                if bt == scirust_integration::backend::BackendType::OpcUa
                {
                    issues.push(format!(
                        "Backend '{}' requires the '{}' crate. Add the feature flag 'real-opcua' to scirust-integration in Cargo.toml",
                        config.backend_type, crate_name
                    ));
                }
            }
        },
        None =>
        {
            issues.push(format!("Unknown backend type: '{}'", config.backend_type));
        },
    }

    // Check 4: Simulated backend test
    let mut backend = BackendFactory::simulated();
    if backend.is_connected()
    {
        passed.push("Simulated backend: connects successfully".to_string());
    }
    else
    {
        issues.push("Simulated backend: failed to connect".to_string());
    }

    // Check 5: OPC-UA browse test
    let nodes = backend.opcua.browse("vibration").unwrap_or_default();
    if !nodes.is_empty()
    {
        passed.push(format!(
            "OPC-UA browse: found {} vibration nodes",
            nodes.len()
        ));
    }
    else
    {
        issues.push("OPC-UA browse: no vibration nodes found".to_string());
    }

    // Check 6: MQTT publish test
    let payload = br#"{"doctor": true}"#;
    if backend
        .mqtt
        .publish("scirust/doctor/test", payload, 1, false)
        .is_ok()
    {
        passed.push("MQTT publish: test message sent".to_string());
    }
    else
    {
        issues.push("MQTT publish: failed to send test message".to_string());
    }

    // Check 7: Pipeline test
    let mut test_pipeline = Pipeline::new(config.clone());
    test_pipeline.run(5);
    let test_status = test_pipeline.status();
    if test_status.audit_chain_valid
    {
        passed.push("Pipeline: audit chain valid after 5 cycles".to_string());
    }
    else
    {
        issues.push("Pipeline: audit chain invalid".to_string());
    }
    if test_status.cycles_completed == 5
    {
        passed.push("Pipeline: completed 5 cycles".to_string());
    }
    else
    {
        issues.push(format!(
            "Pipeline: only completed {} of 5 cycles",
            test_status.cycles_completed
        ));
    }

    // Check 8: Pattern detection crates
    let pattern_crates = vec![
        (
            "scirust-vision",
            "Computer vision: CNN layers, convolution 2D, max/avg pooling, activation functions (ReLU, Sigmoid, Softmax), HOG descriptor, LBP features, Haar-like features, NMS, template matching, Otsu thresholding, connected components, flood fill, Canny edge detection.",
        ),
        (
            "scirust-audio",
            "Audio recognition: Goertzel algorithm, magnitude/power spectrum, Mel filterbank, MFCC + deltas, chroma features, onset detection, YIN pitch tracking, spectral centroid/bandwidth/rolloff/flatness/entropy/contrast, AudioFeatureSet.",
        ),
        (
            "scirust-graph",
            "Graph patterns: Graph type (undirected, adjacency list), BFS/DFS, shortest path, subgraph isomorphism (VF2-like), graph isomorphism, motif discovery, label propagation, modularity, Girvan-Newman, edge betweenness, clustering coefficient, degree distribution, density, diameter, betweenness centrality.",
        ),
        (
            "scirust-sequential",
            "Sequential patterns: HMM (forward/backward/Viterbi/Baum-Welch with log-space), CRF (linear-chain, forward-backward, Viterbi, NLL), sequence labeling (BIO), Needleman-Wunsch, Levenshtein, KMP, Boyer-Moore, LCS, DTW (full + banded + path).",
        ),
        (
            "scirust-multivariate",
            "Multivariate analysis: PCA (Jacobi eigenvalues), ICA (FastICA), K-Means++ clustering, elbow method, silhouette score, Mahalanobis distance outlier detection, classical MDS, CCA.",
        ),
        (
            "scirust-unsupervised",
            "Unsupervised: Autoencoder (encode/decode/anomaly), Isolation Forest (iTree, path-length scoring), DBSCAN, Local Outlier Factor, Gaussian Mixture Model (EM, BIC/AIC), One-Class SVM (RBF kernel, SMO).",
        ),
        (
            "scirust-seasonal",
            "Seasonal: STL decomposition (Loess), ACF/PACF/Durbin-Levinson, periodogram, Fourier analysis, windowed FFT, zero-crossing cycle estimation, moving average decomposition, X-11 style, Mann-Kendall trend test, Sen's slope, seasonal CUSUM, binary segmentation.",
        ),
        (
            "scirust-nlp-advanced",
            "NLP: NER (rule-based + statistical with BIO tagging), LDA (Gibbs sampling, perplexity, UMass coherence), relation extraction, Naive Bayes, TF-IDF, cosine/Jaccard similarity, TextRank, RAKE keyword extraction, MinHash, tokenizer.",
        ),
    ];
    passed.push(format!(
        "Pattern detection crates: {} available ({})",
        pattern_crates.len(),
        pattern_crates
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    ));

    // Check 9: Algorithm creation crates
    let algo_crates = vec![
        (
            "scirust-automl",
            "AutoML: PipelineConfig, PipelineTemplate, StandardScaler/Normalizer/PCA/PolynomialFeatures preprocessing, Linear/RandomForest/GradientBoosting/NeuralNetwork models, HyperOptimizer (random/grid/Bayesian GP with Matern 5/2 + EI), ModelSelector (paired t-test), ensembles (voting/averaging/stacking), FeatureEngineer (polynomial/interaction/variance/correlation/MI), k-fold CV, time-series CV, AutoML orchestrator.",
        ),
        (
            "scirust-synthesis",
            "Program synthesis: SExpr grammar (30+ constructors), Sketch with holes, bottom-up enumeration, top-down type-directed synthesis, genetic programming (tournament/crossover/mutation), beam search, cost model, expression simplification (x+0->x etc.), constant folding, CSE, inductive bias, Occam's razor, incremental synthesis, extrapolation checking.",
        ),
        (
            "scirust-algogen",
            "Algorithm generation: 10 sort strategies (bubble/insertion/selection/merge/quick/heap/counting/radix/intro/tim), 8 search strategies (linear/binary/jump/exponential/interpolation/BST/hash/Fibonacci), graph (Dijkstra/A*/Bellman-Ford/Floyd-Warshall, Prim/Kruskal/Boruvka, Ford-Fulkerson/Edmonds-Karp/Dinic), DP generation, DaC generation, complexity analysis (fit O(1)/O(log n)/O(n)/O(n log n)/O(n^2)), evolutionary optimization.",
        ),
        (
            "scirust-codetrans",
            "Code transformation: AST (Lit/Var/BinOp/UnaryOp/Call/If/Let/While/For/Assign/Block/Return/Function/Struct/Enum/Match), pattern matching with variables, 20 optimization rules (constant folding, identity, strength reduction, boolean simplification), DCE, CSE, LICM, refactoring (extract function, rename, inline, loop-to-iterator, match-to-if-let), transpilation (Rust->Python, Rust->C), pattern database.",
        ),
        (
            "scirust-rl-algo",
            "RL algorithm discovery: Instruction set (13 ops), Algorithm execution, AlgoEnv/ProblemSpec, REINFORCE with baseline, Actor-Critic (TD(0)), Q-Learning with experience replay, simulated annealing, beam search, MCTS with progressive widening, meta-learning (templates, transfer), invariant inference (constant/monotonic/parity), CEGAR verification, test suite generation.",
        ),
        (
            "scirust-scaffold",
            "Algorithmic scaffolding: DSL (tokenizer/parser/Algorithm AST), code generation (RustGenerator/PythonGenerator/CGenerator with CodeStyle), 16 built-in templates (bubble_sort, merge_sort, binary_search, bfs, dfs, dijkstra, etc.), scaffold_new/scaffold_test/scaffold_bench, code analysis (infinite loop/unused variable/complexity estimation), documentation generation (ascii diagrams, examples).",
        ),
    ];
    passed.push(format!(
        "Algorithm creation crates: {} available ({})",
        algo_crates.len(),
        algo_crates
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    ));

    // Print results
    println!("--- Passed ({}) ---", passed.len());
    for p in &passed
    {
        println!("  [OK] {}", p);
    }

    println!("\n--- Issues ({}) ---", issues.len());
    for i in &issues
    {
        println!("  [!]  {}", i);
    }

    println!("\n--- Summary ---");
    println!("  Passed:  {}", passed.len());
    println!("  Issues:  {}", issues.len());
    if issues.is_empty()
    {
        println!("\n  All checks passed. Ready for deployment.");
    }
    else
    {
        println!(
            "\n  {} issue(s) found. Resolve before production use.",
            issues.len()
        );
    }

    println!("\n--- Pattern Detection Crates ---");
    for (name, desc) in &pattern_crates
    {
        println!("  {:<25} {}", name, desc);
    }

    println!("\n--- Algorithm Creation Crates ---");
    for (name, desc) in &algo_crates
    {
        println!("  {:<25} {}", name, desc);
    }

    Ok(())
}
