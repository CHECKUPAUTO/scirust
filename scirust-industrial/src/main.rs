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

mod commands;

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
    };

    if let Err(e) = result
    {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
