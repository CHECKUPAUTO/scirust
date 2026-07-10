//! Mode `--mine` — détection lexicale de « gardes epsilon mortes » dans des
//! bases de code numériques multi-langages (Rust, famille C/CUDA, shaders).
//!
//! ## Question de recherche
//!
//! Une **garde morte** est une constante de garde f32 si petite que la garde
//! ne protège pas. Deux mécanismes de mort, documentés et classés séparément :
//!
//! - **M1 (flush)** : littéral f32 avec `0 < |v| < f32::MIN_POSITIVE`
//!   (`1.17549435e-38`). Un tel littéral est **sous-normal** : sous FTZ/DAZ
//!   (fast-math, drivers GPU, modes CPU) il est écrasé à `0` — `x.max(g)`
//!   devient `x.max(0)` et la garde n'existe plus.
//! - **M2 (inversion)** : littéral f32 avec `0 < |v| < 1/f32::MAX`
//!   (`≈ 2.938736e-39`). Même sans FTZ : si `x.max(g)` vaut `g`, alors
//!   `1.0 / (x.max(g))` déborde au-delà de `f32::MAX` → `inf`. La plage M2
//!   est incluse dans la plage M1 ; un littéral sous `1/f32::MAX` est classé
//!   M2 (mécanisme le plus fort), sinon M1.
//!
//! ## Heuristiques de typage f32 (documentées, lexicales)
//!
//! - **Rust** : suffixe `f32` sur le littéral, ou `f32` présent sur la ligne
//!   → CONFIRMÉ ; suffixe `f64` ou `f64` seul sur la ligne → hors périmètre
//!   (le flottant par défaut de Rust est f64) ; sinon INCERTAIN.
//! - **Famille C / CUDA / OpenCL** : suffixe `f`/`F` → CONFIRMÉ ; littéral nu
//!   sur une ligne contenant `float` → PROBABLE ; sinon hors périmètre ou
//!   INCERTAIN (un littéral nu est un `double` en C — jamais compté comme
//!   finding).
//! - **Shaders (WGSL, GLSL, Metal, compute)** : les flottants sont f32 par
//!   défaut → CONFIRMÉ (et les GPU flushent très couramment les sous-normaux,
//!   ce qui rend M1 d'autant plus probable) ; suffixes `lf` (GLSL double) et
//!   `h` (WGSL f16) → hors périmètre.
//!
//! Seuls CONFIRMÉ et PROBABLE sont des **candidats** ; les INCERTAIN sont
//! listés à part (transparence) et jamais comptés comme findings.
//!
//! ## Limitations (assumées)
//!
//! Parsing purement lexical : pas d'inférence de types sémantique, pas de
//! macro-expansion, pas de littéraux hexadécimaux flottants (`0x1p-40`), pas
//! de constantes construites (`f32::from_bits`), signe non capturé (les
//! gardes sont positives). Chaque candidat exige une **revue manuelle en
//! contexte** avant toute conclusion. L'outil ne modifie jamais les dépôts
//! scannés.

use std::fs;
use std::path::{Path, PathBuf};

/// Seuil M1 : plus petit f32 **normal**. Tout littéral strictement en dessous
/// est sous-normal, donc flushable à 0 sous FTZ/DAZ.
pub const M1_FLUSH_THRESHOLD: f64 = f32::MIN_POSITIVE as f64;

/// Seuil M2 : `1 / f32::MAX`. Tout dénominateur strictement en dessous fait
/// déborder `1.0 / d` en `inf`, même sans FTZ.
pub const M2_INVERSION_THRESHOLD: f64 = 1.0 / (f32::MAX as f64);

/// Extensions de fichiers scannées, avec leur langage.
pub const SCANNED_EXTENSIONS: [(&str, Language); 13] = [
    ("rs", Language::Rust),
    ("c", Language::CFamily),
    ("h", Language::CFamily),
    ("cpp", Language::CFamily),
    ("hpp", Language::CFamily),
    ("cc", Language::CFamily),
    ("cu", Language::CFamily),
    ("cuh", Language::CFamily),
    ("cl", Language::CFamily),
    ("metal", Language::Shader),
    ("wgsl", Language::Shader),
    ("glsl", Language::Shader),
    ("comp", Language::Shader),
];

/// Langage source, pour les règles de commentaires, suffixes et typage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Rust (`.rs`) — suffixes `f32`/`f64`, commentaires de bloc imbriqués.
    Rust,
    /// C, C++, CUDA, OpenCL — suffixe `f`/`F`, littéral nu = `double`.
    CFamily,
    /// WGSL, GLSL, Metal, compute — flottants f32 par défaut.
    Shader,
}

impl Language {
    /// Langage associé à une extension de fichier (insensible à la casse),
    /// ou `None` si l'extension n'est pas scannée.
    pub fn from_extension(ext: &str) -> Option<Language> {
        let lower = ext.to_ascii_lowercase();
        SCANNED_EXTENSIONS
            .iter()
            .find(|(e, _)| *e == lower)
            .map(|(_, l)| *l)
    }

    /// Étiquette courte pour les rapports.
    pub fn label(self) -> &'static str {
        match self
        {
            Language::Rust => "rust",
            Language::CFamily => "c-family",
            Language::Shader => "shader",
        }
    }
}

/// Verdict de typage f32 d'un littéral (heuristique lexicale documentée).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeVerdict {
    /// f32 établi (suffixe explicite, `f32` sur la ligne, ou shader).
    ConfirmedF32,
    /// f32 probable (littéral nu C sur une ligne mentionnant `float`).
    ProbableF32,
    /// Typage indéterminable lexicalement — listé mais jamais compté.
    Uncertain,
    /// Établi non-f32 (suffixe/contexte f64, long double, f16…).
    NotF32,
}

impl TypeVerdict {
    /// Étiquette courte pour les rapports.
    pub fn label(self) -> &'static str {
        match self
        {
            TypeVerdict::ConfirmedF32 => "CONFIRMED-F32",
            TypeVerdict::ProbableF32 => "PROBABLE-F32",
            TypeVerdict::Uncertain => "UNCERTAIN",
            TypeVerdict::NotF32 => "NOT-F32",
        }
    }
}

/// Mécanisme de mort de la garde (M2 ⊂ M1 en plage ; classés séparément).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mechanism {
    /// Sous-normal : flushé à 0 sous FTZ/DAZ.
    M1Flush,
    /// Sous `1/f32::MAX` : `1/d` déborde en `inf` même sans FTZ.
    M2Inversion,
}

impl Mechanism {
    /// Classe une valeur strictement positive déjà connue sous σ. La décision
    /// se prend sur la valeur **arrondie en f32** (sémantique de
    /// matérialisation du littéral dans le programme).
    fn from_value(v: f64) -> Mechanism {
        #[allow(clippy::cast_possible_truncation)]
        let v32 = v as f32;
        if f64::from(v32) < M2_INVERSION_THRESHOLD
        {
            Mechanism::M2Inversion
        }
        else
        {
            Mechanism::M1Flush
        }
    }

    /// Étiquette courte pour les rapports.
    pub fn label(self) -> &'static str {
        match self
        {
            Mechanism::M1Flush => "M1",
            Mechanism::M2Inversion => "M2",
        }
    }
}

/// Marqueurs lexicaux d'un usage de garde sur la ligne (indice de priorité de
/// revue, PAS un filtre : un candidat sans marqueur reste listé). Ordre : du
/// plus spécifique au plus générique — le premier trouvé est retenu.
const GUARD_MARKERS: [&str; 12] = [
    ".max(", "fmaxf", "fmax", "max(", "clamp", "denom", "eps", "abs", "sqrt", "log", "/", "<",
];

/// Un littéral sous-seuil repéré dans une source, avant revue manuelle.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Chemin relatif à la racine minée.
    pub file: PathBuf,
    /// Numéro de ligne (1-based).
    pub line: usize,
    /// Texte brut du littéral (sans suffixe), ex. `1e-40`.
    pub literal: String,
    /// Valeur numérique (positive).
    pub value: f64,
    pub language: Language,
    pub mechanism: Mechanism,
    pub verdict: TypeVerdict,
    /// Premier marqueur de garde trouvé sur la ligne, s'il y en a un.
    pub guard_marker: Option<&'static str>,
    /// Ligne source rognée (≤ 160 caractères) pour la revue.
    pub extract: String,
}

/// Un drapeau fast-math/FTZ repéré dans un fichier de build.
#[derive(Debug, Clone)]
pub struct FastMathHit {
    pub file: PathBuf,
    pub line: usize,
    /// Motif ayant déclenché (`-ffast-math`, `use_fast_math`, …).
    pub pattern: &'static str,
}

/// Résultat complet du minage d'un répertoire.
#[derive(Debug, Default)]
pub struct MineOutcome {
    /// Candidats à revue : verdicts CONFIRMÉ ou PROBABLE.
    pub candidates: Vec<Candidate>,
    /// Littéraux sous-seuil au typage indéterminable — jamais comptés.
    pub uncertain: Vec<Candidate>,
    /// Littéraux sous-seuil établis non-f32 (comptés seulement).
    pub not_f32_count: usize,
    /// Drapeaux fast-math/FTZ dans les fichiers de build.
    pub fastmath: Vec<FastMathHit>,
    /// Fichiers source effectivement scannés.
    pub files_scanned: usize,
    /// Lignes source effectivement scannées.
    pub lines_scanned: usize,
    /// Fichiers écartés par les règles d'exclusion (tests, vendor…).
    pub files_excluded: usize,
}

// =========================================================================
// Exclusions de chemins
// =========================================================================

/// Répertoires exclus par préfixe de composant (minuscules).
const EXCLUDED_DIR_PREFIXES: [&str; 3] = ["test", "benchmark", "bench"];

/// Répertoires exclus par égalité exacte de composant (minuscules).
const EXCLUDED_DIR_EXACT: [&str; 7] = [
    "third_party",
    "3rdparty",
    "vendor",
    "external",
    "target",
    "build",
    "node_modules",
];

/// Vrai si le chemin relatif relève d'un contexte exclu du minage :
/// répertoires `test*/`, `bench*/`, `benchmark*/`, `third_party/`, `vendor/`,
/// `external/`, artefacts (`target/`, `build/`), répertoires cachés, et
/// fichiers `*_test.*` / `test_*.*`.
pub fn is_excluded_path(rel: &Path) -> bool {
    let mut components = rel.components().peekable();
    while let Some(comp) = components.next()
    {
        let name = comp.as_os_str().to_string_lossy().to_ascii_lowercase();
        let is_last = components.peek().is_none();
        if is_last
        {
            // Composant final = nom de fichier.
            let stem = name.split('.').next().unwrap_or(&name);
            if stem.ends_with("_test") || stem.ends_with("_tests") || stem.starts_with("test_")
            {
                return true;
            }
        }
        else
        {
            if name.starts_with('.')
            {
                return true;
            }
            if EXCLUDED_DIR_EXACT.contains(&name.as_str())
            {
                return true;
            }
            if EXCLUDED_DIR_PREFIXES.iter().any(|p| name.starts_with(p))
            {
                return true;
            }
        }
    }
    false
}

// =========================================================================
// Scanner lexical multi-langage
// =========================================================================

/// Suffixe de littéral, interprété selon le langage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuffixKind {
    None,
    F32,
    NotF32,
    Other,
}

fn classify_suffix(lang: Language, suffix: &str) -> SuffixKind {
    if suffix.is_empty()
    {
        return SuffixKind::None;
    }
    match lang
    {
        Language::Rust => match suffix
        {
            "f32" => SuffixKind::F32,
            "f64" => SuffixKind::NotF32,
            _ => SuffixKind::Other,
        },
        Language::CFamily => match suffix
        {
            "f" | "F" => SuffixKind::F32,
            "l" | "L" => SuffixKind::NotF32,
            _ => SuffixKind::Other,
        },
        Language::Shader => match suffix
        {
            "f" | "F" => SuffixKind::F32,
            // GLSL `lf` = double ; WGSL `h` = f16.
            "lf" | "LF" | "h" => SuffixKind::NotF32,
            _ => SuffixKind::Other,
        },
    }
}

/// Verdict de typage d'un littéral nu (sans suffixe décisif) selon sa ligne.
fn bare_literal_verdict(lang: Language, line: &str) -> TypeVerdict {
    match lang
    {
        Language::Rust =>
        {
            let has32 = line.contains("f32");
            let has64 = line.contains("f64");
            match (has32, has64)
            {
                (true, false) => TypeVerdict::ConfirmedF32,
                (false, true) => TypeVerdict::NotF32,
                // Mixte → indéterminable ; ni f32 ni f64 → le défaut Rust est
                // f64 par inférence, mais la ligne ne suffit pas : INCERTAIN.
                (true, true) | (false, false) => TypeVerdict::Uncertain,
            }
        },
        Language::CFamily =>
        {
            if line.contains("double")
            {
                TypeVerdict::NotF32
            }
            else if line.contains("float")
            {
                TypeVerdict::ProbableF32
            }
            else
            {
                // Littéral nu en C = double : jamais compté comme finding.
                TypeVerdict::Uncertain
            }
        },
        Language::Shader => TypeVerdict::ConfirmedF32,
    }
}

fn prev_is_ident(chars: &[char], i: usize) -> bool {
    if i == 0
    {
        return false;
    }
    matches!(chars.get(i - 1), Some(p) if p.is_ascii_alphanumeric() || *p == '_' || *p == '.')
}

/// Littéral brut issu du scanner : position, texte, valeur, suffixe.
struct RawLiteral {
    line: usize,
    text: String,
    value: f64,
    suffix: SuffixKind,
}

/// Extrait d'une source les littéraux flottants strictement positifs et
/// strictement sous [`M1_FLUSH_THRESHOLD`], hors commentaires (`//`, `/* */` —
/// imbriqués en Rust seulement), chaînes et littéraux de caractère.
fn scan_subthreshold_literals(lang: Language, src: &str) -> Vec<RawLiteral> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;

    while i < n
    {
        let c = match chars.get(i)
        {
            Some(c) => *c,
            None => break,
        };

        if c == '\n'
        {
            line += 1;
            i += 1;
            continue;
        }

        // Commentaire de ligne.
        if c == '/' && matches!(chars.get(i + 1), Some('/'))
        {
            while i < n && !matches!(chars.get(i), Some('\n'))
            {
                i += 1;
            }
            continue;
        }

        // Commentaire de bloc (imbriqué en Rust, plat ailleurs).
        if c == '/' && matches!(chars.get(i + 1), Some('*'))
        {
            let mut depth = 1usize;
            i += 2;
            while i < n && depth > 0
            {
                if matches!(chars.get(i), Some('\n'))
                {
                    line += 1;
                    i += 1;
                }
                else if lang == Language::Rust
                    && matches!(chars.get(i), Some('/'))
                    && matches!(chars.get(i + 1), Some('*'))
                {
                    depth += 1;
                    i += 2;
                }
                else if matches!(chars.get(i), Some('*')) && matches!(chars.get(i + 1), Some('/'))
                {
                    depth -= 1;
                    i += 2;
                }
                else
                {
                    i += 1;
                }
            }
            continue;
        }

        // Chaîne brute Rust : r"…" / r#"…"#.
        if lang == Language::Rust
            && c == 'r'
            && !prev_is_ident(&chars, i)
            && matches!(chars.get(i + 1), Some('"') | Some('#'))
        {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while matches!(chars.get(j), Some('#'))
            {
                hashes += 1;
                j += 1;
            }
            if matches!(chars.get(j), Some('"'))
            {
                j += 1;
                loop
                {
                    match chars.get(j)
                    {
                        None => break,
                        Some('\n') =>
                        {
                            line += 1;
                            j += 1;
                        },
                        Some('"') =>
                        {
                            let mut k = j + 1;
                            let mut cnt = 0usize;
                            while cnt < hashes && matches!(chars.get(k), Some('#'))
                            {
                                cnt += 1;
                                k += 1;
                            }
                            if cnt == hashes
                            {
                                j = k;
                                break;
                            }
                            j += 1;
                        },
                        Some(_) => j += 1,
                    }
                }
                i = j;
                continue;
            }
        }

        // Chaîne classique "…" avec échappements.
        if c == '"'
        {
            i += 1;
            while i < n
            {
                match chars.get(i)
                {
                    Some('\\') =>
                    {
                        if matches!(chars.get(i + 1), Some('\n'))
                        {
                            line += 1;
                        }
                        i += 2;
                    },
                    Some('\n') =>
                    {
                        line += 1;
                        i += 1;
                    },
                    Some('"') =>
                    {
                        i += 1;
                        break;
                    },
                    Some(_) => i += 1,
                    None => break,
                }
            }
            continue;
        }

        // Littéral de caractère '…' (Rust : distinguer les lifetimes).
        if c == '\'' && lang != Language::Shader
        {
            if matches!(chars.get(i + 1), Some('\\'))
            {
                i += 2;
                while i < n && !matches!(chars.get(i), Some('\'') | Some('\n'))
                {
                    i += 1;
                }
                if matches!(chars.get(i), Some('\''))
                {
                    i += 1;
                }
            }
            else if matches!(chars.get(i + 2), Some('\''))
            {
                i += 3;
            }
            else
            {
                // Lifetime Rust ('a, 'static) ou apostrophe isolée.
                i += 1;
            }
            continue;
        }

        // Début possible d'un littéral numérique : chiffre, ou `.chiffre`
        // (hors Rust, où `.5` n'est pas un littéral).
        let digit_start = c.is_ascii_digit() && !prev_is_ident(&chars, i);
        let dot_start = lang != Language::Rust
            && c == '.'
            && matches!(chars.get(i + 1), Some(d) if d.is_ascii_digit())
            && !prev_is_ident(&chars, i);
        if digit_start || dot_start
        {
            let start = i;
            let start_line = line;
            let mut is_float = dot_start;

            while matches!(chars.get(i), Some(d) if d.is_ascii_digit() || *d == '_')
            {
                i += 1;
            }
            // Partie fractionnaire : `.` suivi d'au moins un chiffre — ou
            // point de départ `.5` déjà détecté.
            if matches!(chars.get(i), Some('.'))
                && matches!(chars.get(i + 1), Some(d) if d.is_ascii_digit())
            {
                is_float = true;
                i += 1;
                while matches!(chars.get(i), Some(d) if d.is_ascii_digit() || *d == '_')
                {
                    i += 1;
                }
            }
            // Exposant : e/E, signe optionnel, au moins un chiffre.
            if matches!(chars.get(i), Some('e') | Some('E'))
            {
                let mut j = i + 1;
                if matches!(chars.get(j), Some('+') | Some('-'))
                {
                    j += 1;
                }
                if matches!(chars.get(j), Some(d) if d.is_ascii_digit())
                {
                    is_float = true;
                    i = j;
                    while matches!(chars.get(i), Some(d) if d.is_ascii_digit() || *d == '_')
                    {
                        i += 1;
                    }
                }
            }

            let value_end = i;
            // Suffixe éventuel (f32, f, F, l, lf, h, …).
            while matches!(chars.get(i), Some(s) if s.is_ascii_alphanumeric() || *s == '_')
            {
                i += 1;
            }
            let suffix_text: String = chars.get(value_end..i).unwrap_or(&[]).iter().collect();
            let suffix = classify_suffix(lang, &suffix_text);
            if suffix == SuffixKind::F32 || suffix == SuffixKind::NotF32
            {
                is_float = true;
            }

            if is_float && suffix != SuffixKind::Other
            {
                let raw: String = chars.get(start..value_end).unwrap_or(&[]).iter().collect();
                // Le `_` séparateur Rust avant suffixe (`1e-38_f32`) reste
                // collé au texte : on le retire de l'affichage et du parse.
                let text = raw.trim_end_matches('_').to_string();
                let cleaned: String = text.chars().filter(|ch| *ch != '_').collect();
                if let Ok(v) = cleaned.parse::<f64>()
                {
                    // Sémantique de matérialisation : le littéral vivra en
                    // f32 dans le programme — c'est sa valeur ARRONDIE en f32
                    // qui décide s'il est sous σ. (Ex. : `1.17549435e-38`
                    // parse en f64 juste SOUS le σ exact, mais arrondit en
                    // f32 exactement À `f32::MIN_POSITIVE` — garde licite.)
                    #[allow(clippy::cast_possible_truncation)]
                    let v32 = v as f32;
                    if v.is_finite() && v > 0.0 && v32 < f32::MIN_POSITIVE
                    {
                        out.push(RawLiteral {
                            line: start_line,
                            text,
                            value: v,
                            suffix,
                        });
                    }
                }
            }
            continue;
        }

        i += 1;
    }

    out
}

/// Rogne une ligne pour l'extrait (≤ 160 caractères, coupure sûre UTF-8).
fn trim_extract(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.chars().count() <= 160
    {
        return trimmed.to_string();
    }
    let cut = trimmed
        .char_indices()
        .nth(160)
        .map(|(b, _)| b)
        .unwrap_or(trimmed.len());
    let head = trimmed.get(..cut).unwrap_or(trimmed);
    format!("{head}…")
}

/// Scanne une source et classe chaque littéral sous-seuil. Cœur pur (sans
/// E/S) : c'est la cible des tests unitaires sur fixtures synthétiques.
pub fn scan_source(lang: Language, rel: &Path, src: &str) -> Vec<Candidate> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for lit in scan_subthreshold_literals(lang, src)
    {
        let line_text = lines.get(lit.line.saturating_sub(1)).copied().unwrap_or("");
        let verdict = match lit.suffix
        {
            SuffixKind::F32 => TypeVerdict::ConfirmedF32,
            SuffixKind::NotF32 => TypeVerdict::NotF32,
            SuffixKind::None => bare_literal_verdict(lang, line_text),
            // Suffixe inconnu (ex. `u64`) : littéral non flottant — déjà
            // écarté par le scanner, branche par complétude.
            SuffixKind::Other => TypeVerdict::NotF32,
        };
        let guard_marker = GUARD_MARKERS
            .iter()
            .find(|m| line_text.contains(**m))
            .copied();
        out.push(Candidate {
            file: rel.to_path_buf(),
            line: lit.line,
            literal: lit.text,
            value: lit.value,
            language: lang,
            mechanism: Mechanism::from_value(lit.value),
            verdict,
            guard_marker,
            extract: trim_extract(line_text),
        });
    }
    out
}

// =========================================================================
// Drapeaux fast-math / FTZ dans les fichiers de build
// =========================================================================

/// Motifs cherchés dans les fichiers de build. `use_fast_math` (sans tirets)
/// couvre `--use_fast_math` (nvcc) et `-use_fast_math`. `ftz` est cherché
/// insensiblement à la casse (`-ftz=true`, `FTZ`, `__CUDA_FTZ`…).
const FASTMATH_PATTERNS: [&str; 3] = [
    "-ffast-math",
    "use_fast_math",
    "-funsafe-math-optimizations",
];

/// Vrai si le nom de fichier désigne un fichier de build à inspecter.
pub fn is_build_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("cmakelists")
        || lower.starts_with("makefile")
        || lower == "build.rs"
        || lower == "setup.py"
        || lower == "meson.build"
        || lower == "build"
    {
        return true;
    }
    matches!(
        lower.rsplit('.').next(),
        Some("cmake") | Some("mk") | Some("bzl") | Some("bazel") | Some("gn") | Some("gni")
    )
}

/// Cherche les drapeaux fast-math/FTZ dans le contenu d'un fichier de build.
pub fn scan_build_file(rel: &Path, contents: &str) -> Vec<FastMathHit> {
    let mut out = Vec::new();
    for (idx, line) in contents.lines().enumerate()
    {
        for pat in FASTMATH_PATTERNS
        {
            if line.contains(pat)
            {
                out.push(FastMathHit {
                    file: rel.to_path_buf(),
                    line: idx + 1,
                    pattern: pat,
                });
            }
        }
        if line.to_ascii_lowercase().contains("ftz")
        {
            out.push(FastMathHit {
                file: rel.to_path_buf(),
                line: idx + 1,
                pattern: "ftz",
            });
        }
    }
    out
}

// =========================================================================
// Parcours du système de fichiers
// =========================================================================

/// Parcourt `dir` récursivement ; les erreurs d'E/S sont signalées sur stderr
/// sans interrompre la campagne (un dépôt partiellement lisible reste miné).
fn walk(root: &Path, dir: &Path, outcome: &mut MineOutcome) {
    let entries = match fs::read_dir(dir)
    {
        Ok(e) => e,
        Err(e) =>
        {
            eprintln!(
                "epsilon-audit --mine: avertissement : lecture de {} impossible : {e}",
                dir.display()
            );
            return;
        },
    };
    // Tri des entrées pour un parcours (et un rapport) déterministe.
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries
    {
        match entry
        {
            Ok(e) => paths.push(e.path()),
            Err(e) =>
            {
                eprintln!(
                    "epsilon-audit --mine: avertissement : entrée illisible dans {} : {e}",
                    dir.display()
                );
            },
        }
    }
    paths.sort();

    for path in paths
    {
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        if path.is_dir()
        {
            // Les exclusions de répertoires s'appliquent au chemin relatif
            // augmenté d'un composant fictif (le test regarde les répertoires).
            if is_excluded_path(&rel.join("_"))
            {
                continue;
            }
            walk(root, &path, outcome);
            continue;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Fichiers de build : scannés pour fast-math même hors extensions
        // sources (mais pas sous les répertoires exclus).
        if is_build_file(&name) && !is_excluded_path(&rel)
        {
            if let Ok(contents) = fs::read_to_string(&path)
            {
                outcome.fastmath.extend(scan_build_file(&rel, &contents));
            }
        }
        let ext = match path.extension().map(|e| e.to_string_lossy().into_owned())
        {
            Some(e) => e,
            None => continue,
        };
        let lang = match Language::from_extension(&ext)
        {
            Some(l) => l,
            None => continue,
        };
        if is_excluded_path(&rel)
        {
            outcome.files_excluded += 1;
            continue;
        }
        let src = match fs::read_to_string(&path)
        {
            // Fichier non-UTF-8 (binaire, encodage exotique) : ignoré.
            Ok(s) => s,
            Err(_) => continue,
        };
        outcome.files_scanned += 1;
        outcome.lines_scanned += src.lines().count();
        for cand in scan_source(lang, &rel, &src)
        {
            match cand.verdict
            {
                TypeVerdict::ConfirmedF32 | TypeVerdict::ProbableF32 =>
                {
                    outcome.candidates.push(cand);
                },
                TypeVerdict::Uncertain => outcome.uncertain.push(cand),
                TypeVerdict::NotF32 => outcome.not_f32_count += 1,
            }
        }
    }
}

/// Mine un répertoire : candidats, incertains, drapeaux fast-math, stats.
/// Le résultat est trié (fichier, ligne) — déterministe pour un même arbre.
pub fn mine_dir(root: &Path) -> MineOutcome {
    let mut outcome = MineOutcome::default();
    walk(root, root, &mut outcome);
    let key = |c: &Candidate| (c.file.clone(), c.line);
    outcome.candidates.sort_by_key(key);
    outcome.uncertain.sort_by_key(key);
    outcome
        .fastmath
        .sort_by_key(|h| (h.file.clone(), h.line, h.pattern));
    outcome
}

// =========================================================================
// Tests unitaires — fixtures synthétiques exigées par la mission
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(lang: Language, src: &str) -> Vec<Candidate> {
        scan_source(lang, Path::new("fixture"), src)
    }

    // --- Seuils : cohérence avec la bibliothèque σ ---

    #[test]
    fn thresholds_match_mission_values() {
        // M1 = f32::MIN_POSITIVE = 1.17549435e-38 ; M2 = 1/f32::MAX ≈ 2.938736e-39.
        // Vérifié à la COMPILATION (const) : les seuils de la mission sont
        // exactement ceux du module.
        const { assert!((M1_FLUSH_THRESHOLD - 1.17549435e-38).abs() < 1e-45) };
        const { assert!((M2_INVERSION_THRESHOLD - 2.938736e-39).abs() < 1e-45) };
        const { assert!(M2_INVERSION_THRESHOLD < M1_FLUSH_THRESHOLD) };
    }

    // --- Fixture 1 : vrai dead guard M1 (Rust, suffixe f32, garde .max) ---

    #[test]
    fn rust_m1_dead_guard_is_confirmed() {
        let cands = scan(Language::Rust, "let d = x.max(1e-38_f32);\n");
        assert_eq!(cands.len(), 1);
        let c = match cands.first()
        {
            Some(c) => c,
            None => return,
        };
        assert_eq!(c.verdict, TypeVerdict::ConfirmedF32);
        assert_eq!(c.mechanism, Mechanism::M1Flush);
        assert_eq!(c.guard_marker, Some(".max("));
        assert_eq!(c.line, 1);
        assert_eq!(c.literal, "1e-38");
    }

    // --- Fixture 2 : vrai dead guard M2 (C, suffixe f, fmaxf + division) ---

    #[test]
    fn c_m2_dead_guard_is_confirmed() {
        let src = "float g = fmaxf(x, 1e-40f);\ny = 1.0f / g;\n";
        let cands = scan(Language::CFamily, src);
        assert_eq!(cands.len(), 1);
        let c = match cands.first()
        {
            Some(c) => c,
            None => return,
        };
        assert_eq!(c.verdict, TypeVerdict::ConfirmedF32);
        assert_eq!(c.mechanism, Mechanism::M2Inversion);
        assert_eq!(c.guard_marker, Some("fmaxf"));
    }

    // --- Fixture 3 : f64 bénin → jamais candidat ---

    #[test]
    fn rust_f64_literal_is_not_a_candidate() {
        let cands = scan(Language::Rust, "let eps = 1e-40_f64;\n");
        assert_eq!(cands.len(), 1);
        assert!(matches!(
            cands.first().map(|c| c.verdict),
            Some(TypeVerdict::NotF32)
        ));
    }

    #[test]
    fn c_double_literal_is_not_a_candidate() {
        let cands = scan(Language::CFamily, "double eps = 1e-300;\n");
        // 1e-300 est hors plage f32 (< M1 en valeur, mais la ligne dit double).
        assert!(matches!(
            cands.first().map(|c| c.verdict),
            Some(TypeVerdict::NotF32)
        ));
    }

    #[test]
    fn c_bare_literal_without_float_is_uncertain() {
        // Littéral nu en C = double : INCERTAIN, jamais compté comme finding.
        let cands = scan(Language::CFamily, "eps = 1e-40;\n");
        assert!(matches!(
            cands.first().map(|c| c.verdict),
            Some(TypeVerdict::Uncertain)
        ));
    }

    #[test]
    fn c_bare_literal_with_float_on_line_is_probable() {
        let cands = scan(Language::CFamily, "const float eps = 1e-39;\n");
        assert_eq!(cands.len(), 1);
        let c = match cands.first()
        {
            Some(c) => c,
            None => return,
        };
        assert_eq!(c.verdict, TypeVerdict::ProbableF32);
        assert_eq!(c.mechanism, Mechanism::M2Inversion);
    }

    // --- Fixture 4 : contexte test → exclu par le chemin ---

    #[test]
    fn test_paths_are_excluded() {
        assert!(is_excluded_path(Path::new("tests/foo.rs")));
        assert!(is_excluded_path(Path::new("src/tests/foo.c")));
        assert!(is_excluded_path(Path::new("src/foo_test.cc")));
        assert!(is_excluded_path(Path::new("src/test_ops.cu")));
        assert!(is_excluded_path(Path::new("benchmarks/bench.cpp")));
        assert!(is_excluded_path(Path::new("third_party/lib/x.h")));
        assert!(is_excluded_path(Path::new("vendor/x.rs")));
        assert!(is_excluded_path(Path::new("external/x.glsl")));
        assert!(!is_excluded_path(Path::new("src/ops/softmax.cu")));
        assert!(!is_excluded_path(Path::new("kernel/attest.rs")));
    }

    // --- Shaders : f32 par défaut ---

    #[test]
    fn shader_bare_literal_is_confirmed_f32() {
        let cands = scan(Language::Shader, "let g = max(x, 1e-40);\n");
        assert_eq!(cands.len(), 1);
        let c = match cands.first()
        {
            Some(c) => c,
            None => return,
        };
        assert_eq!(c.verdict, TypeVerdict::ConfirmedF32);
        assert_eq!(c.mechanism, Mechanism::M2Inversion);
        assert_eq!(c.guard_marker, Some("max("));
    }

    // --- Bornes de plage : une garde valide n'est pas capturée ---

    #[test]
    fn literal_at_or_above_sigma_is_ignored() {
        // 1.17549435e-38 == f32::MIN_POSITIVE : garde licite, hors plage.
        let cands = scan(Language::Rust, "let d = x.max(1.17549435e-38_f32);\n");
        assert!(cands.is_empty());
        let cands = scan(Language::Rust, "let d = x.max(1e-30_f32);\n");
        assert!(cands.is_empty());
    }

    #[test]
    fn m1_vs_m2_boundary_is_respected() {
        // 5e-39 ∈ [1/f32::MAX, f32::MIN_POSITIVE) → M1 seul.
        let cands = scan(Language::Rust, "let d = x.max(5e-39f32);\n");
        assert!(matches!(
            cands.first().map(|c| c.mechanism),
            Some(Mechanism::M1Flush)
        ));
        // 1e-39 < 1/f32::MAX → M2.
        let cands = scan(Language::Rust, "let d = x.max(1e-39f32);\n");
        assert!(matches!(
            cands.first().map(|c| c.mechanism),
            Some(Mechanism::M2Inversion)
        ));
    }

    // --- Commentaires et chaînes : jamais scannés ---

    #[test]
    fn literals_in_comments_and_strings_are_ignored() {
        let src = "// garde morte : 1e-40f32\n/* 1e-40f32 */\nlet s = \"1e-40f32\";\n";
        assert!(scan(Language::Rust, src).is_empty());
        let src = "// 1e-40f\n/* 1e-40f */\nchar* s = \"1e-40f\";\n";
        assert!(scan(Language::CFamily, src).is_empty());
    }

    // --- C : littéral à point initial (.5e-39f) ---

    #[test]
    fn c_leading_dot_literal_is_scanned() {
        let cands = scan(Language::CFamily, "float g = .5e-39f;\n");
        assert_eq!(cands.len(), 1);
        assert!(matches!(
            cands.first().map(|c| c.verdict),
            Some(TypeVerdict::ConfirmedF32)
        ));
    }

    // --- Fast-math : détection dans les fichiers de build ---

    #[test]
    fn fastmath_flags_are_detected_in_build_files() {
        assert!(is_build_file("CMakeLists.txt"));
        assert!(is_build_file("Makefile.am"));
        assert!(is_build_file("build.rs"));
        assert!(is_build_file("setup.py"));
        assert!(is_build_file("rules.cmake"));
        assert!(!is_build_file("main.cpp"));

        let hits = scan_build_file(
            Path::new("CMakeLists.txt"),
            "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} --use_fast_math\")\n\
             add_compile_options(-ffast-math)\n\
             # FTZ enabled by driver\n",
        );
        let patterns: Vec<&str> = hits.iter().map(|h| h.pattern).collect();
        assert!(patterns.contains(&"use_fast_math"));
        assert!(patterns.contains(&"-ffast-math"));
        assert!(patterns.contains(&"ftz"));
    }

    // --- Déterminisme du tri de mine_dir (sur les candidats en mémoire) ---

    #[test]
    fn extract_is_bounded() {
        let long_line = format!("let d = x.max(1e-40f32); // {}", "x".repeat(400));
        let cands = scan(Language::Rust, &long_line);
        assert!(
            cands
                .first()
                .is_some_and(|c| c.extract.chars().count() <= 161)
        );
    }
}
