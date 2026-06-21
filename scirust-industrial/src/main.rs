//! SciRust Industrial CLI
//!
//! Facilitates integration of SciRust with real industrial systems.
//!
//! Commands:
//!   discover    — Browse OPC-UA server for available sensor nodes
//!   test-opcua  — Test OPC-UA connection and read values
//!   test-mqtt   — Test MQTT broker connection
//!   gen-config  — Generate a pipeline configuration file
//!   scaffold    — Generate a complete monitoring project
//!   run         — Run a monitoring pipeline from config
//!   doctor      — Diagnose integration issues
//!
//! Vertical demos (deterministic, run against the real crate API):
//!   nav-tdoa     — TDOA emitter multilateration
//!   nav-fusion   — GNSS/INS fusion with a GNSS outage
//!   track-imm    — Interacting Multiple Models tracking
//!   track-ud     — UD square-root Kalman filter vs textbook Kalman
//!   water-leak   — acoustic leak localization
//!   water-surge  — water-hammer surge (Joukowsky / Korteweg)
//!   ot-firmware  — firmware attestation
//!   ot-plc       — PLC ladder integrity + Stuxnet write-set
//!   golden-batch — GMP golden-batch comparator (21 CFR Part 11)

mod commands;
mod verticals;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "scirust-industrial")]
#[command(about = "SciRust Industrial Integration CLI")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Browse OPC-UA server for available sensor nodes
    Discover {
        /// OPC-UA endpoint URL
        #[arg(long, default_value = "opc.tcp://localhost:4840")]
        endpoint: String,
        /// Filter pattern for node names
        #[arg(long, default_value = "")]
        filter: String,
        /// Use simulated backend (ignore endpoint)
        #[arg(long)]
        simulated: bool,
    },

    /// Test OPC-UA connection and read sensor values
    TestOpcua {
        #[arg(long, default_value = "opc.tcp://localhost:4840")]
        endpoint: String,
        #[arg(long)]
        simulated: bool,
        /// Number of samples to read
        #[arg(long, default_value_t = 5)]
        samples: usize,
    },

    /// Test MQTT broker connection
    TestMqtt {
        #[arg(long, default_value = "localhost")]
        host: String,
        #[arg(long, default_value_t = 1883)]
        port: u16,
        #[arg(long)]
        simulated: bool,
        /// Test message to publish
        #[arg(long, default_value = "scirust/test/hello")]
        topic: String,
    },

    /// Generate a pipeline configuration file
    GenConfig {
        /// Output file path
        #[arg(long, default_value = "config.json")]
        output: String,
        /// Configuration template: minimal, automotive, bearing, pdm
        #[arg(long, default_value = "automotive")]
        template: String,
        /// Number of stations (for automotive template)
        #[arg(long, default_value_t = 3)]
        stations: usize,
        /// Line identifier
        #[arg(long, default_value = "line-1")]
        line_id: String,
    },

    /// Generate a complete monitoring project
    Scaffold {
        /// Project name
        #[arg(long)]
        name: String,
        /// Output directory
        #[arg(long, default_value = ".")]
        output: String,
        /// Template: minimal, automotive, bearing, pdm
        #[arg(long, default_value = "minimal")]
        template: String,
    },

    /// Run a monitoring pipeline from config
    Run {
        /// Config file path
        #[arg(long, default_value = "config.json")]
        config: String,
        /// Number of cycles to run
        #[arg(long, default_value_t = 100)]
        cycles: usize,
        /// Output report to JSON file
        #[arg(long)]
        report: Option<String>,
    },

    /// Diagnose integration issues
    Doctor {
        /// Config file to check
        #[arg(long, default_value = "config.json")]
        config: String,
    },

    /// Navigation: locate an emitter by time-difference-of-arrival (TDOA)
    NavTdoa {
        /// Wave speed (m/s)
        #[arg(long, default_value_t = 1500.0)]
        speed: f64,
    },

    /// Navigation: loosely-coupled GNSS/INS fusion with a GNSS outage
    NavFusion {
        /// Number of time steps
        #[arg(long, default_value_t = 60)]
        steps: usize,
        /// GNSS outage length in steps (applied mid-run)
        #[arg(long, default_value_t = 10)]
        outage: usize,
    },

    /// Estimation: IMM filter shifts onto the maneuver model on a maneuver
    TrackImm {
        #[arg(long, default_value_t = 120)]
        steps: usize,
    },

    /// Estimation: UD square-root Kalman filter vs textbook Kalman
    TrackUd {
        #[arg(long, default_value_t = 80)]
        steps: usize,
    },

    /// Water: locate a leak by acoustic cross-correlation
    WaterLeak {
        #[arg(long, default_value_t = 100.0)]
        pipe_length: f64,
        #[arg(long, default_value_t = 1000.0)]
        wave_speed: f64,
        #[arg(long, default_value_t = 10000.0)]
        sample_rate: f64,
        /// True leak distance from sensor A (m)
        #[arg(long, default_value_t = 30.0)]
        leak_at: f64,
    },

    /// Water: water-hammer surge (Joukowsky) and wave speed (Korteweg)
    WaterSurge {
        #[arg(long, default_value_t = 1000.0)]
        rho: f64,
        #[arg(long, default_value_t = 1200.0)]
        wave_speed: f64,
        #[arg(long, default_value_t = 2.0)]
        delta_v: f64,
        #[arg(long, default_value_t = 2.2e9)]
        bulk: f64,
        #[arg(long, default_value_t = 200e9)]
        e_pipe: f64,
        #[arg(long, default_value_t = 0.5)]
        diameter: f64,
        #[arg(long, default_value_t = 0.01)]
        wall: f64,
    },

    /// OT security: firmware attestation (clean vs tampered image)
    OtFirmware {
        /// Firmware size in bytes
        #[arg(long, default_value_t = 4096)]
        size: usize,
        /// Block size in bytes
        #[arg(long, default_value_t = 256)]
        block: usize,
        /// Which block to corrupt in the tampered image
        #[arg(long, default_value_t = 3)]
        tamper_block: usize,
    },

    /// OT security: PLC ladder integrity + Stuxnet write-set detection
    OtPlc {},

    /// GMP: golden-batch comparator (DTW align + hash-chained audit)
    GoldenBatch {
        /// Phase lag (steps) prepended to the candidate batch
        #[arg(long, default_value_t = 10)]
        lag: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command
    {
        Commands::Discover {
            endpoint,
            filter,
            simulated,
        } => commands::discover(endpoint, filter, *simulated),
        Commands::TestOpcua {
            endpoint,
            simulated,
            samples,
        } => commands::test_opcua(endpoint, *simulated, *samples),
        Commands::TestMqtt {
            host,
            port,
            simulated,
            topic,
        } => commands::test_mqtt(host, *port, *simulated, topic),
        Commands::GenConfig {
            output,
            template,
            stations,
            line_id,
        } => commands::gen_config(output, template, *stations, line_id),
        Commands::Scaffold {
            name,
            output,
            template,
        } => commands::scaffold(name, output, template),
        Commands::Run {
            config,
            cycles,
            report,
        } => commands::run(config, *cycles, report.as_deref()),
        Commands::Doctor { config } => commands::doctor(config),
        Commands::NavTdoa { speed } => verticals::nav_tdoa(*speed),
        Commands::NavFusion { steps, outage } => verticals::nav_fusion(*steps, *outage),
        Commands::TrackImm { steps } => verticals::track_imm(*steps),
        Commands::TrackUd { steps } => verticals::track_ud(*steps),
        Commands::WaterLeak {
            pipe_length,
            wave_speed,
            sample_rate,
            leak_at,
        } => verticals::water_leak(*pipe_length, *wave_speed, *sample_rate, *leak_at),
        Commands::WaterSurge {
            rho,
            wave_speed,
            delta_v,
            bulk,
            e_pipe,
            diameter,
            wall,
        } => verticals::water_surge(
            *rho,
            *wave_speed,
            *delta_v,
            *bulk,
            *e_pipe,
            *diameter,
            *wall,
        ),
        Commands::OtFirmware {
            size,
            block,
            tamper_block,
        } => verticals::ot_firmware(*size, *block, *tamper_block),
        Commands::OtPlc {} => verticals::ot_plc(),
        Commands::GoldenBatch { lag } => verticals::golden_batch(*lag),
    };

    if let Err(e) = result
    {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
