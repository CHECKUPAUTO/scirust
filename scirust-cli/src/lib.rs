//! `scirust` — one entry point for the whole toolkit.
//!
//! A thin, discoverable dispatcher over capabilities that already exist and
//! are tested elsewhere in the workspace: it adds no new compute, only a
//! command surface so users don't have to hand-write the library API for
//! common tasks. `scirust help` lists everything; `scirust info` describes
//! the guarantees.

pub mod learning;
pub mod nlp;
pub mod numeric;
pub mod quickstart;
pub mod reasoning;
pub mod symbolic;
pub mod synergy;
pub mod trader;

/// One registered command for the help listing.
struct Command {
    name: &'static str,
    args: &'static str,
    about: &'static str,
}

/// Commands grouped by theme, in display order.
const GROUPS: &[(&str, &[Command])] = &[
    (
        "LEARNING & OPTIMIZATION",
        &[
            Command {
                name: "quickstart",
                args: "",
                about: "Train the XOR demo MLP (deterministic) end to end → 4/4.",
            },
            Command {
                name: "som train",
                args: "[--seed N] [--epochs E]",
                about: "Train the ownership model; report accuracy vs baseline.",
            },
            Command {
                name: "evo",
                args: "[--seed N] [--gens G]",
                about: "Minimize the sphere function with a seeded genetic algorithm.",
            },
            Command {
                name: "cmaes",
                args: "[--seed N] [--steps S]",
                about: "Minimize the sphere function with a seeded CMA-ES.",
            },
        ],
    ),
    (
        "SYMBOLIC MATH",
        &[
            Command {
                name: "diff",
                args: "<expr> [var]",
                about: "Symbolic derivative, e.g. `diff \"x^2 + 3*x\"`.",
            },
            Command {
                name: "simplify",
                args: "<expr>",
                about: "Algebraic simplification of an expression.",
            },
            Command {
                name: "eval",
                args: "<expr> [x=.. ..]",
                about: "Evaluate an expression at given variable values.",
            },
            Command {
                name: "solve",
                args: "<expr> [var]",
                about: "Symbolic real roots of `expr = 0` (linear / quadratic).",
            },
            Command {
                name: "prove",
                args: "<a> <b>",
                about: "Best-effort proof that two expressions are equivalent.",
            },
            Command {
                name: "gradient",
                args: "<expr> x=.. [y=..]",
                about: "Numeric gradient at a point (1 or 2 variables).",
            },
            Command {
                name: "to-rust",
                args: "<expr>",
                about: "Transpile an expression to Rust source.",
            },
            Command {
                name: "regress",
                args: "<xs> <ys> [degree]",
                about: "Least-squares fit (linear, or polynomial of given degree).",
            },
            Command {
                name: "symreg",
                args: "<xs> <ys> [--seed N]",
                about: "Discover a closed-form y=f(x) by genetic programming.",
            },
            Command {
                name: "trig",
                args: "<expr>",
                about: "Apply trigonometric identities, then simplify.",
            },
            Command {
                name: "patterns",
                args: "\"v1,v2,..\"",
                about: "Detect trend patterns in a numeric series.",
            },
        ],
    ),
    (
        "LOGIC",
        &[Command {
            name: "sat",
            args: "\"1,-2;2;-1,3\"",
            about: "DPLL satisfiability (clauses ; literals , ; -v = ¬v).",
        }],
    ),
    (
        "NUMERICAL SOLVERS",
        &[
            Command {
                name: "pinn",
                args: "[--seed N] [--steps S]",
                about: "Physics-Informed NN: solve u''=-u (BVP) with the PDE residual in the loss; checks vs sin x.",
            },
            Command {
                name: "integrate",
                args: "<expr> <a> <b> [var] [--method M]",
                about: "Definite integral (Romberg | simpson | gauss).",
            },
            Command {
                name: "root",
                args: "<expr> <a> <b> [var] [--method M]",
                about: "A root in [a,b] (brent | bisection; needs a sign change).",
            },
            Command {
                name: "minimize",
                args: "<expr> <a> <b> [var]",
                about: "Local minimum in [a,b] (root of the symbolic derivative).",
            },
            Command {
                name: "optimize",
                args: "<expr> --start a,b --vars x,y",
                about: "Multi-variable minimum via Nelder–Mead (derivative-free).",
            },
            Command {
                name: "linsolve",
                args: "\"r;r\" \"b\"",
                about: "Solve A·x = b by LU, e.g. `linsolve \"2,1;1,3\" \"3,5\"`.",
            },
            Command {
                name: "lstsq",
                args: "\"r;r;r\" \"b\"",
                about: "Least-squares A·x ≈ b via QR (rows ≥ cols).",
            },
            Command {
                name: "det",
                args: "\"r;r\"",
                about: "Determinant of a square matrix.",
            },
            Command {
                name: "cholesky",
                args: "\"r;r\"",
                about: "Cholesky factor L of an SPD matrix (A = L·Lᵀ).",
            },
            Command {
                name: "qr",
                args: "\"r;r\"",
                about: "QR decomposition A = Q·R (prints Q and R).",
            },
            Command {
                name: "cg",
                args: "\"r;r\" \"b\"",
                about: "Solve SPD A·x = b by conjugate gradient (iterative).",
            },
            Command {
                name: "inverse",
                args: "\"r;r\"",
                about: "Inverse of a square matrix (LU).",
            },
            Command {
                name: "solve-system",
                args: "\"f1;f2\" --vars x,y --start a,b",
                about: "Solve a nonlinear system F(x)=0 (Broyden quasi-Newton).",
            },
            Command {
                name: "polyroots",
                args: "\"c0,c1,..\"",
                about: "Real roots of a polynomial (ascending coefficients).",
            },
            Command {
                name: "ode",
                args: "<f(t,y)> <y0> <t0> <t1> [h] [--method M]",
                about: "Integrate dy/dt = f(t,y) (rk4 | dopri5 adaptive).",
            },
            Command {
                name: "fem-heat",
                args: "<nodes> <length> <source>",
                about: "1D steady heat -u''=source via linear finite elements.",
            },
        ],
    ),
    (
        "TENSOR NETWORKS",
        &[Command {
            name: "tt",
            args: "\"r;r\" [--factors d] [--max-rank r] [--tol t] [--max-err e]",
            about: "Tensor-train (TT-SVD) compression of a matrix; reports ratio & error.",
        }],
    ),
    (
        "NLP",
        &[
            Command {
                name: "bpe",
                args: "\"<corpus>\" [--vocab N] [--encode \"<text>\"] [--bytes]",
                about: "Train a deterministic BPE tokenizer (--bytes = lossless byte-level).",
            },
            Command {
                name: "lm",
                args: "[\"t0,t1,..\"] [--seed N] [--steps S] [--lr R] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]",
                about: "Train a tiny causal decoder LM (N-D tape) to recall a token sequence.",
            },
            Command {
                name: "deltanet",
                args: "[--seed N] [--steps S]",
                about: "Train a single-head DeltaNet (delta-rule linear attention) layer to fit a sequence.",
            },
            Command {
                name: "mamba",
                args: "[--seed N] [--steps S]",
                about: "Train a Mamba selective state-space layer (S6 scan) to fit a sequence.",
            },
            Command {
                name: "retnet",
                args: "[--seed N] [--steps S]",
                about: "Train a RetNet retention layer (linear attention, recurrent ≡ parallel) to fit a sequence.",
            },
            Command {
                name: "gla",
                args: "[--seed N] [--steps S]",
                about: "Train a Gated Linear Attention layer (data-dependent forget gate) to fit a sequence.",
            },
            Command {
                name: "hgrn",
                args: "[--seed N] [--steps S]",
                about: "Train an HGRN gated-linear-RNN token mixer (lower-bounded forget gate) to fit a sequence.",
            },
            Command {
                name: "rwkv",
                args: "[--seed N] [--steps S]",
                about: "Train an RWKV time-mixing (WKV) layer (per-channel decay + bonus) to fit a sequence.",
            },
        ],
    ),
    (
        "CODE ANALYSIS",
        &[Command {
            name: "analyze",
            args: "<file.rs> [--sarif]",
            about: "Ownership analysis of real Rust (use-after-move, borrows). SARIF for CI.",
        }],
    ),
    (
        "INFERENCE INTEGRITY",
        &[
            Command {
                name: "verify",
                args: "emit|verify <args..>",
                about: "Emit or check a deterministic inference proof certificate.",
            },
            Command {
                name: "certify",
                args: "[--seed N] [--eps E]",
                about: "Certify a ReLU MLP's output bounds over an L∞ box via interval propagation (IBP).",
            },
            Command {
                name: "conformal",
                args: "[--seed N] [--alpha A]",
                about: "Split-conformal prediction intervals with a distribution-free coverage guarantee.",
            },
            Command {
                name: "calibrate",
                args: "[--seed N]",
                about: "Temperature scaling: fit T to lower the expected calibration error (accuracy unchanged).",
            },
        ],
    ),
    (
        "COMPRESSION",
        &[
            Command {
                name: "gptq",
                args: "[--seed N] [--samples S] [--damp D]",
                about: "GPTQ int8 weight quantization (error feedback); reports the error reduction vs round-to-nearest.",
            },
            Command {
                name: "awq",
                args: "[--seed N] [--samples S] [--grid G]",
                about: "AWQ activation-aware int8 quantization (search-based per-channel scaling); reports the error reduction vs round-to-nearest.",
            },
            Command {
                name: "bitnet",
                args: "[--seed N]",
                about: "BitNet b1.58 ternary {-1,0,+1} quantization; verifies the multiplication-free matmul.",
            },
        ],
    ),
    (
        "META",
        &[
            Command {
                name: "info",
                args: "",
                about: "Capabilities, guarantees, determinism.",
            },
            Command {
                name: "help",
                args: "",
                about: "Show this list of commands.",
            },
            Command {
                name: "version",
                args: "",
                about: "Print the scirust CLI version.",
            },
        ],
    ),
    (
        "CRYPTO TRADING (AUDITABLE)",
        &[Command {
            name: "trader",
            args: "run|predict|audit|info [args]",
            about: "Auditable crypto-trading pipeline: certified predictions, LLM narration, proof-sealed decisions.",
        }],
    ),
    (
        "PATTERN DETECTION CRATES",
        &[
            Command {
                name: "scirust-vision",
                args: "",
                about: "Computer vision: CNN, conv2D, pooling, HOG, LBP, Haar-like, NMS, template matching, Otsu, Canny.",
            },
            Command {
                name: "scirust-audio",
                args: "",
                about: "Audio: Goertzel, spectrum, Mel filterbank, MFCC+deltas, chroma, onset, YIN pitch, spectral features.",
            },
            Command {
                name: "scirust-graph",
                args: "",
                about: "Graph: BFS/DFS, shortest path, subgraph isomorphism (VF2), motif discovery, label propagation, modularity, centrality.",
            },
            Command {
                name: "scirust-sequential",
                args: "",
                about: "Sequential: HMM forward/backward/Viterbi/Baum-Welch, CRF, Needleman-Wunsch, DTW (full+banded+path).",
            },
            Command {
                name: "scirust-multivariate",
                args: "",
                about: "Multivariate: PCA (Jacobi), ICA (FastICA), K-Means++, silhouette, Mahalanobis, MDS, CCA.",
            },
            Command {
                name: "scirust-unsupervised",
                args: "",
                about: "Unsupervised: Autoencoder, Isolation Forest, DBSCAN, LOF, GMM (EM, BIC/AIC), One-Class SVM.",
            },
            Command {
                name: "scirust-seasonal",
                args: "",
                about: "Seasonal: STL, ACF/PACF, periodogram, X-11, Mann-Kendall, Sen's slope, seasonal CUSUM, binary segmentation.",
            },
            Command {
                name: "scirust-nlp-advanced",
                args: "",
                about: "NLP: NER (BIO), LDA (Gibbs), relation extraction, TF-IDF, TextRank, RAKE, MinHash, tokenizer.",
            },
        ],
    ),
    (
        "ALGORITHM CREATION CRATES",
        &[
            Command {
                name: "scirust-automl",
                args: "",
                about: "AutoML: preprocessing, model selection, hyperparameter optimization (Bayesian GP, grid, random), ensembles.",
            },
            Command {
                name: "scirust-synthesis",
                args: "",
                about: "Program synthesis: SExpr grammar, Sketch, bottom-up/top-down enumeration, genetic programming, beam search.",
            },
            Command {
                name: "scirust-algogen",
                args: "",
                about: "Algorithm generation: 10 sorts, 8 searches, graph algos, DP, DaC, complexity fitting, evolutionary optimization.",
            },
            Command {
                name: "scirust-codetrans",
                args: "",
                about: "Code transformation: AST, pattern matching, 20 opt rules, DCE, CSE, LICM, refactoring, Rust->Python/C transpilation.",
            },
            Command {
                name: "scirust-rl-algo",
                args: "",
                about: "RL algorithm discovery: REINFORCE, Actor-Critic, Q-Learning, MCTS, meta-learning, CEGAR, invariant inference.",
            },
            Command {
                name: "scirust-scaffold",
                args: "",
                about: "Algorithmic scaffolding: DSL, code gen (Rust/Python/C), 16 templates, code analysis, doc generation.",
            },
        ],
    ),
];

fn print_help() {
    println!("scirust — pure-Rust deterministic ML & scientific-computing toolkit\n");
    println!("usage: scirust <command> [args]\n");
    let width = GROUPS
        .iter()
        .flat_map(|(_, cs)| cs.iter())
        .map(|c| c.name.len() + c.args.len() + 1)
        .max()
        .unwrap_or(0);
    for (group, cmds) in GROUPS
    {
        println!("{group}");
        for c in *cmds
        {
            let sig = if c.args.is_empty()
            {
                c.name.to_string()
            }
            else
            {
                format!("{} {}", c.name, c.args)
            };
            println!("  {sig:<width$}  {}", c.about);
        }
        println!();
    }
    println!("run a command with no further args for its specific usage.");
}

fn print_info() {
    println!(
        "scirust {} — pure Rust, zero FFI\n",
        env!("CARGO_PKG_VERSION")
    );
    println!("Guarantees:");
    println!("  • Deterministic: seeded PCG RNG everywhere; same seed ⇒ bit-identical output.");
    println!("  • Oracle-validated: every numeric primitive is tested against a reference.");
    println!("  • Stable Rust: the whole workspace builds and tests on stable (nightly only");
    println!("    for the optional `portable-simd` feature).");
    println!(
        "  • Auditable: pure Rust, no C/C++/Python, Cargo.lock committed, cargo-deny in CI.\n"
    );
    println!("Highlights reachable from this CLI:");
    println!("  • Deep-learning core + reverse-mode autodiff (`quickstart`).");
    println!("  • Ownership analysis of real Rust source (`analyze`, `som train`).");
    println!("  • Symbolic math: differentiation, simplification, solving (`diff`/`solve`/…).");
    println!("  • Evolutionary optimization (`evo`).");
    println!("  • Verifiable, reproducible inference certificates (`verify`).");
    println!("  • Pattern detection: 8 crates for vision, audio, graph, sequential, multivariate,");
    println!(
        "    unsupervised, seasonal, and NLP analysis (`scirust-vision`, `scirust-audio`, …)."
    );
    println!(
        "  • Algorithm creation: 6 crates for AutoML, program synthesis, algorithm generation,"
    );
    println!("    code transformation, RL discovery, and scaffolding (`scirust-automl`, …).\n");
    println!("Docs: README.md · docs/REFERENCE.md · `cargo doc --workspace --no-deps --open`");
}

/// Dispatch `args` (excluding the program name). Returns the exit code.
pub fn run(args: &[String]) -> u8 {
    let rest = if args.len() > 1 { &args[1..] } else { &[] };
    match args.first().map(String::as_str)
    {
        None | Some("help") | Some("-h") | Some("--help") =>
        {
            print_help();
            0
        },
        Some("version") | Some("--version") | Some("-V") =>
        {
            println!("scirust {}", env!("CARGO_PKG_VERSION"));
            0
        },
        Some("info") =>
        {
            print_info();
            0
        },
        Some("quickstart") => quickstart::run(),
        Some("som") => learning::run_som(rest),
        Some("certify") => learning::run_certify(rest),
        Some("conformal") => learning::run_conformal(rest),
        Some("calibrate") => learning::run_calibrate(rest),
        Some("kvcache") => synergy::run_kvcache(rest),
        Some("guard") => synergy::run_guard(rest),
        Some("attest") => synergy::run_attest(rest),
        Some("pinn") => learning::run_pinn(rest),
        Some("quantum") => learning::run_quantum(rest),
        Some("gptq") => learning::run_gptq(rest),
        Some("awq") => learning::run_awq(rest),
        Some("bitnet") => learning::run_bitnet(rest),
        Some("evo") => learning::run_evo(rest),
        Some("cmaes") => learning::run_cmaes(rest),
        Some("diff") => symbolic::run_diff(rest),
        Some("simplify") => symbolic::run_simplify(rest),
        Some("eval") => symbolic::run_eval(rest),
        Some("solve") => symbolic::run_solve(rest),
        Some("prove") => symbolic::run_prove(rest),
        Some("gradient") => symbolic::run_gradient(rest),
        Some("to-rust") => symbolic::run_to_rust(rest),
        Some("regress") => symbolic::run_regress(rest),
        Some("symreg") => reasoning::run_symreg(rest),
        Some("trig") => symbolic::run_trig(rest),
        Some("patterns") => symbolic::run_patterns(rest),
        Some("sat") => reasoning::run_sat(rest),
        Some("integrate") => numeric::run_integrate(rest),
        Some("root") => numeric::run_root(rest),
        Some("minimize") => numeric::run_minimize(rest),
        Some("optimize") => numeric::run_optimize(rest),
        Some("linsolve") => numeric::run_linsolve(rest),
        Some("lstsq") => numeric::run_lstsq(rest),
        Some("det") => numeric::run_det(rest),
        Some("cholesky") => numeric::run_cholesky(rest),
        Some("qr") => numeric::run_qr(rest),
        Some("cg") => numeric::run_cg(rest),
        Some("polyroots") => numeric::run_polyroots(rest),
        Some("ode") => numeric::run_ode(rest),
        Some("inverse") => numeric::run_inverse(rest),
        Some("solve-system") => numeric::run_solve_system(rest),
        Some("fem-heat") => numeric::run_fem_heat(rest),
        Some("tt") => numeric::run_tt(rest),
        Some("bpe") => nlp::run_bpe(rest),
        Some("lm") => nlp::run_lm(rest),
        Some("deltanet") => nlp::run_deltanet(rest),
        Some("mamba") => nlp::run_mamba(rest),
        Some("retnet") => nlp::run_retnet(rest),
        Some("gla") => nlp::run_gla(rest),
        Some("hgrn") => nlp::run_hgrn(rest),
        Some("rwkv") => nlp::run_rwkv(rest),
        Some("analyze") => scirust_som_cli::run(rest, "scirust analyze"),
        Some("verify") => scirust_runtime::proofcli::run(rest),
        Some("trader") => trader::run(rest),
        Some(other) =>
        {
            eprintln!("unknown command: `{other}`\n");
            print_help();
            2
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn meta_commands_succeed() {
        assert_eq!(run(&[]), 0);
        assert_eq!(run(&s(&["help"])), 0);
        assert_eq!(run(&s(&["version"])), 0);
        assert_eq!(run(&s(&["info"])), 0);
    }

    #[test]
    fn unknown_command_is_rejected() {
        assert_eq!(run(&s(&["frobnicate"])), 2);
    }

    #[test]
    fn dispatch_reaches_each_group() {
        assert_eq!(run(&s(&["quickstart"])), 0);
        assert_eq!(run(&s(&["diff", "x*x"])), 0);
        assert_eq!(run(&s(&["solve", "x^2 - 4"])), 0);
        assert_eq!(run(&s(&["evo", "--gens", "20"])), 0);
        assert_eq!(run(&s(&["cmaes", "--steps", "20"])), 0);
        assert_eq!(run(&s(&["som", "train", "--epochs", "3"])), 0);
        assert_eq!(run(&s(&["to-rust", "x + 1"])), 0);
        assert_eq!(run(&s(&["regress", "0,1,2", "1,3,5"])), 0);
        assert_eq!(run(&s(&["integrate", "x", "0", "1"])), 0);
        assert_eq!(run(&s(&["root", "x^2 - 2", "0", "2"])), 0);
        assert_eq!(run(&s(&["minimize", "x^2 - 4*x + 7", "0", "5"])), 0);
        assert_eq!(run(&s(&["linsolve", "2,1;1,3", "3,5"])), 0);
        assert_eq!(run(&s(&["det", "2,1;1,3"])), 0);
        assert_eq!(run(&s(&["polyroots", "-2,0,1"])), 0);
        assert_eq!(run(&s(&["ode", "y", "1", "0", "1"])), 0);
        assert_eq!(run(&s(&["prove", "x + x", "2*x"])), 0);
        assert_eq!(run(&s(&["gradient", "x^2", "x=3"])), 0);
        assert_eq!(run(&s(&["lstsq", "1,0;1,1;1,2", "1,2,3"])), 0);
        assert_eq!(run(&s(&["cholesky", "4,2;2,3"])), 0);
        assert_eq!(
            run(&s(&["root", "x^2 - 2", "0", "2", "--method", "secant"])),
            0
        );
        assert_eq!(
            run(&s(&["root", "x^2 - 2", "0", "2", "--method", "newton"])),
            0
        );
        assert_eq!(run(&s(&["sat", "1,-2;2"])), 0);
        assert_eq!(run(&s(&["sat", "1;-1"])), 1);
        assert_eq!(run(&s(&["symreg", "0,1,2,3", "0,2,4,6"])), 0);
        assert_eq!(run(&s(&["trig", "sin(x)^2 + cos(x)^2"])), 0);
        assert_eq!(run(&s(&["patterns", "1,2,3,4"])), 0);
        assert_eq!(run(&s(&["qr", "1,1;0,1;1,0"])), 0);
        assert_eq!(run(&s(&["cg", "4,1;1,3", "1,2"])), 0);
        assert_eq!(run(&s(&["inverse", "4,7;2,6"])), 0);
        assert_eq!(
            run(&s(&[
                "solve-system",
                "x^2 + y^2 - 4; x - y",
                "--vars",
                "x,y",
                "--start",
                "1,1"
            ])),
            0
        );
        assert_eq!(run(&s(&["fem-heat", "9", "1", "1"])), 0);
        assert_eq!(
            run(&s(&["ode", "y", "1", "0", "1", "--method", "dopri5"])),
            0
        );
        assert_eq!(run(&s(&["tt", "1,2,3,4;2,4,6,8;3,6,9,12;4,8,12,16"])), 0);
        assert_eq!(run(&s(&["bpe", "low lower lowest", "--vocab", "30"])), 0);
        assert_eq!(run(&s(&["lm", "1,2,3,1,2,3", "--steps", "10"])), 0);
        assert_eq!(run(&s(&["deltanet", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["mamba", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["retnet", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["gla", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["hgrn", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["rwkv", "--steps", "5"])), 0);
        assert_eq!(run(&s(&["certify", "--eps", "0.02"])), 0);
        assert_eq!(run(&s(&["conformal", "--alpha", "0.1"])), 0);
        assert_eq!(run(&s(&["calibrate", "--seed", "1"])), 0);
        assert_eq!(run(&s(&["pinn", "--steps", "50"])), 0);
        assert_eq!(run(&s(&["gptq", "--seed", "1"])), 0);
        assert_eq!(run(&s(&["awq", "--seed", "1"])), 0);
        assert_eq!(run(&s(&["bitnet", "--seed", "1"])), 0);
        assert_eq!(
            run(&s(&[
                "optimize",
                "(x-1)^2 + (y-2)^2",
                "--vars",
                "x,y",
                "--start",
                "0,0"
            ])),
            0
        );
    }

    #[test]
    fn usage_errors_return_two() {
        assert_eq!(run(&s(&["analyze"])), 2);
        assert_eq!(run(&s(&["verify"])), 2);
        assert_eq!(run(&s(&["diff"])), 2);
        assert_eq!(run(&s(&["eval"])), 2);
    }

    #[test]
    fn trader_subcommands_work() {
        assert_eq!(run(&s(&["trader", "info"])), 0);
        assert_eq!(
            run(&s(&[
                "trader",
                "run",
                "--steps",
                "2",
                "--output",
                "/tmp/scirust_cli_trader_test.json"
            ])),
            0
        );
        assert_eq!(
            run(&s(&[
                "trader",
                "audit",
                "/tmp/scirust_cli_trader_test.json"
            ])),
            0
        );
        assert_eq!(run(&s(&["trader", "predict"])), 0);
    }
}
