//! `epsilon-audit` — audit lexical des littéraux epsilon du workspace SciRust.
//!
//! Outil **std-only** (le hachage SHA-256 optionnel du rapport passe par `sha2`,
//! déjà présent au lockfile, sous le drapeau `report-hash`). Aucun `regex`,
//! aucun `syn`, aucun `serde`. Le parsing est un scanner lexical maison qui
//! détecte les littéraux flottants (`\d+e-?\d+`, `\d+\.\d+e-?\d+`, `\d+\.\d+`)
//! hors commentaires et chaînes.
//!
//! ## Catégories (heuristiques, documentées)
//!
//! - **A — algorithmique** : constante d'un algorithme publié (`epsilon:`,
//!   `eps:`, `self.epsilon`, `weight_decay`, `beta`). NE PAS migrer.
//! - **B — garde contre zéro** : `.max(`, `/ (` (dénominateur), `abs() <`,
//!   `< f32::EPSILON`, `< f64::EPSILON`. Cible de migration future vers σ.
//! - **C — test/tolérance** : sous `tests/`/`examples/`/`benches/`, ou ligne
//!   contenant `assert`, ou après un `#[cfg(test)]` / `mod tests`.
//! - **D — convergence** : `tol`, `while`, `for` + comparaison. Risque de
//!   reproductibilité (dépendant d'échelle).
//! - **U — non classé** : catégorie honnête, aucun classement forcé.
//!
//! ## Mode `--check` (gate CI)
//!
//! Sort avec un code ≠ 0 s'il existe, HORS catégorie C, un littéral **f32** de
//! valeur `> 0` et `< σ_sanitized` (= [`scirust_sigma::SIGMA_SANITIZED_F32`])
//! dans les sources de `scirust-gpu/src/` — la voie sanitized, où une telle
//! garde est morte. Un littéral est réputé f32 s'il porte le suffixe `f32` ou
//! si sa ligne mentionne `f32` ; toute ambiguïté f32/f64 est signalée en
//! WARNING (jamais bloquante — zéro faux positif).
//!
//! ## Mode `--mine <dir>` (campagne externe)
//!
//! Mine un dépôt **externe** (multi-langage : Rust, C/C++/CUDA/OpenCL,
//! shaders) à la recherche de gardes epsilon mortes — littéraux f32 sous
//! `f32::MIN_POSITIVE` (mécanisme M1, flush FTZ/DAZ) ou sous `1/f32::MAX`
//! (mécanisme M2, inversion en `inf`). Voir `scirust_sigma::mine` pour les
//! heuristiques exactes. Sortie : rapport Markdown déterministe (+ bloc TSV
//! pour agrégation). Ce mode est purement analytique : code de sortie 0, ne
//! modifie jamais le dépôt scanné.
//!
//! Le binaire ne modifie JAMAIS aucun fichier.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use scirust_sigma::SIGMA_SANITIZED_F32;
use scirust_sigma::mine;

const EXIT_OK: u8 = 0;
const EXIT_CHECK_FAILED: u8 = 1;
const EXIT_IO: u8 = 2;
const EXIT_USAGE: u8 = 3;

/// Répertoires ignorés lors du parcours.
const SKIP_DIRS: [&str; 3] = ["target", "archive", ".git"];

// =========================================================================
// Erreurs
// =========================================================================

enum AuditError {
    Io(io::Error),
    Usage(String),
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            AuditError::Io(e) => write!(f, "erreur d'E/S : {e}"),
            AuditError::Usage(m) => write!(f, "{m}"),
        }
    }
}

impl From<io::Error> for AuditError {
    fn from(e: io::Error) -> Self {
        AuditError::Io(e)
    }
}

// =========================================================================
// Catégories
// =========================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Category {
    Algorithmic,
    Guard,
    Test,
    Convergence,
    Unclassified,
}

impl Category {
    /// Ordre stable pour les tableaux (A, B, C, D, U).
    const ALL: [Category; 5] = [
        Category::Algorithmic,
        Category::Guard,
        Category::Test,
        Category::Convergence,
        Category::Unclassified,
    ];

    fn index(self) -> usize {
        match self
        {
            Category::Algorithmic => 0,
            Category::Guard => 1,
            Category::Test => 2,
            Category::Convergence => 3,
            Category::Unclassified => 4,
        }
    }

    fn code(self) -> char {
        match self
        {
            Category::Algorithmic => 'A',
            Category::Guard => 'B',
            Category::Test => 'C',
            Category::Convergence => 'D',
            Category::Unclassified => 'U',
        }
    }

    fn label(self) -> &'static str {
        match self
        {
            Category::Algorithmic => "A — constante d'algorithme (ne pas migrer)",
            Category::Guard => "B — garde contre zéro (cible σ)",
            Category::Test => "C — test / tolérance",
            Category::Convergence => "D — seuil de convergence",
            Category::Unclassified => "U — non classé",
        }
    }
}

// =========================================================================
// Modèle de données
// =========================================================================

/// Verdict f32 d'un littéral, pour le gate `--check`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum F32Verdict {
    Yes,
    No,
    Ambiguous,
}

/// Un littéral flottant < 1.0 repéré dans une source.
struct Finding {
    /// Chemin relatif à la racine auditée.
    file: PathBuf,
    /// Numéro de ligne (1-based).
    line: usize,
    /// Texte brut du littéral (sans suffixe), ex. `1e-8`.
    text: String,
    /// Valeur numérique.
    value: f64,
    category: Category,
    f32_verdict: F32Verdict,
    /// Extrait de la ligne (rognée), pour l'annexe.
    extract: String,
}

/// Un littéral brut issu du scanner (avant classification).
struct RawLiteral {
    line: usize,
    text: String,
    value: f64,
    suffix_f32: bool,
    suffix_f64: bool,
}

// =========================================================================
// Scanner lexical
// =========================================================================

fn prev_is_ident(chars: &[char], i: usize) -> bool {
    if i == 0
    {
        return false;
    }
    let p = chars[i - 1];
    p.is_ascii_alphanumeric() || p == '_'
}

/// Extrait les littéraux flottants `< 1.0` d'une source, hors commentaires
/// (ligne `//`, bloc `/* */` imbriqués), chaînes (`"…"`, brutes `r#"…"#`) et
/// littéraux de caractère (`'…'`). Les numéros de ligne sont 1-based.
fn scan_float_literals(src: &str) -> Vec<RawLiteral> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;

    while i < n
    {
        let c = chars[i];

        if c == '\n'
        {
            line += 1;
            i += 1;
            continue;
        }

        // Commentaire de ligne : // … jusqu'à la fin de ligne.
        if c == '/' && i + 1 < n && chars[i + 1] == '/'
        {
            while i < n && chars[i] != '\n'
            {
                i += 1;
            }
            continue;
        }

        // Commentaire de bloc imbriqué : /* … */.
        if c == '/' && i + 1 < n && chars[i + 1] == '*'
        {
            let mut depth = 1usize;
            i += 2;
            while i < n && depth > 0
            {
                if chars[i] == '\n'
                {
                    line += 1;
                    i += 1;
                }
                else if chars[i] == '/' && i + 1 < n && chars[i + 1] == '*'
                {
                    depth += 1;
                    i += 2;
                }
                else if chars[i] == '*' && i + 1 < n && chars[i + 1] == '/'
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

        // Chaîne brute : r"…" ou r#…"…"#… (le 'r' ne doit pas suivre un ident).
        if c == 'r'
            && !prev_is_ident(&chars, i)
            && i + 1 < n
            && (chars[i + 1] == '"' || chars[i + 1] == '#')
        {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < n && chars[j] == '#'
            {
                hashes += 1;
                j += 1;
            }
            if j < n && chars[j] == '"'
            {
                j += 1;
                loop
                {
                    if j >= n
                    {
                        break;
                    }
                    if chars[j] == '\n'
                    {
                        line += 1;
                        j += 1;
                        continue;
                    }
                    if chars[j] == '"'
                    {
                        let mut k = j + 1;
                        let mut cnt = 0usize;
                        while k < n && cnt < hashes && chars[k] == '#'
                        {
                            cnt += 1;
                            k += 1;
                        }
                        if cnt == hashes
                        {
                            j = k;
                            break;
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }

        // Chaîne classique : "…" avec échappements.
        if c == '"'
        {
            i += 1;
            while i < n
            {
                if chars[i] == '\\'
                {
                    if i + 1 < n && chars[i + 1] == '\n'
                    {
                        line += 1;
                    }
                    i += 2;
                    continue;
                }
                if chars[i] == '\n'
                {
                    line += 1;
                    i += 1;
                    continue;
                }
                if chars[i] == '"'
                {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Littéral de caractère '…' vs lifetime 'a.
        if c == '\''
        {
            if i + 1 < n && chars[i + 1] == '\\'
            {
                i += 2;
                while i < n && chars[i] != '\'' && chars[i] != '\n'
                {
                    i += 1;
                }
                if i < n && chars[i] == '\''
                {
                    i += 1;
                }
                continue;
            }
            else if i + 2 < n && chars[i + 2] == '\''
            {
                i += 3;
                continue;
            }
            else
            {
                // Lifetime ('a, 'static, …) — avancer d'un seul caractère.
                i += 1;
                continue;
            }
        }

        // Début possible d'un littéral numérique.
        if c.is_ascii_digit() && !prev_is_ident(&chars, i)
        {
            let start = i;
            let start_line = line;
            let mut is_float = false;

            while i < n && (chars[i].is_ascii_digit() || chars[i] == '_')
            {
                i += 1;
            }
            // Partie fractionnaire : `.` suivi d'au moins un chiffre.
            if i < n && chars[i] == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()
            {
                is_float = true;
                i += 1;
                while i < n && (chars[i].is_ascii_digit() || chars[i] == '_')
                {
                    i += 1;
                }
            }
            // Exposant : e/E, signe optionnel, au moins un chiffre.
            if i < n && (chars[i] == 'e' || chars[i] == 'E')
            {
                let mut j = i + 1;
                if j < n && (chars[j] == '+' || chars[j] == '-')
                {
                    j += 1;
                }
                if j < n && chars[j].is_ascii_digit()
                {
                    is_float = true;
                    i = j;
                    while i < n && (chars[i].is_ascii_digit() || chars[i] == '_')
                    {
                        i += 1;
                    }
                }
            }

            let value_end = i;
            // Suffixe éventuel (f32/f64/i32/…).
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_')
            {
                i += 1;
            }
            let suffix: String = chars[value_end..i].iter().collect();
            let suffix_f32 = suffix == "f32";
            let suffix_f64 = suffix == "f64";
            if suffix_f32 || suffix_f64
            {
                is_float = true;
            }

            if is_float
            {
                let text: String = chars[start..value_end].iter().collect();
                let cleaned: String = text.chars().filter(|ch| *ch != '_').collect();
                if let Ok(v) = cleaned.parse::<f64>()
                {
                    if v.is_finite() && v < 1.0
                    {
                        out.push(RawLiteral {
                            line: start_line,
                            text,
                            value: v,
                            suffix_f32,
                            suffix_f64,
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

// =========================================================================
// Classification
// =========================================================================

/// Vrai si le chemin relève d'un contexte de test par son emplacement.
fn path_is_test(rel: &Path) -> bool {
    let s = rel.to_string_lossy();
    for marker in ["tests/", "examples/", "benches/"]
    {
        if s.starts_with(marker) || s.contains(&format!("/{marker}"))
        {
            return true;
        }
    }
    false
}

/// Détermine, ligne par ligne, si un marqueur `#[cfg(test)]` / `mod tests`
/// précède (latch simple, sans AST).
fn test_active_flags(lines: &[&str]) -> Vec<bool> {
    let mut flags = Vec::with_capacity(lines.len());
    let mut active = false;
    for l in lines
    {
        if l.contains("#[cfg(test)]") || l.contains("mod tests")
        {
            active = true;
        }
        flags.push(active);
    }
    flags
}

fn classify(in_test_dir: bool, line_text: &str, test_active: bool) -> Category {
    // C — test : emplacement, `assert`, ou après un marqueur de test.
    if in_test_dir || test_active || line_text.contains("assert")
    {
        return Category::Test;
    }
    // A — algorithmique : champ/défaut nommé.
    if line_text.contains("epsilon:")
        || line_text.contains("eps:")
        || line_text.contains("self.epsilon")
        || line_text.contains("weight_decay")
        || line_text.contains("beta")
    {
        return Category::Algorithmic;
    }
    // B — garde contre zéro.
    if line_text.contains(".max(")
        || line_text.contains("/ (")
        || line_text.contains("abs() <")
        || line_text.contains("< f32::EPSILON")
        || line_text.contains("< f64::EPSILON")
    {
        return Category::Guard;
    }
    // D — convergence.
    if line_text.contains("tol")
        || line_text.contains("while")
        || (line_text.contains("for ") && (line_text.contains('<') || line_text.contains('>')))
    {
        return Category::Convergence;
    }
    Category::Unclassified
}

/// Verdict f32 : suffixe explicite, sinon mention sur la ligne, sinon ambigu/non.
fn f32_verdict(lit: &RawLiteral, line_text: &str) -> F32Verdict {
    if lit.suffix_f32
    {
        return F32Verdict::Yes;
    }
    if lit.suffix_f64
    {
        return F32Verdict::No;
    }
    let line_f32 = line_text.contains("f32");
    let line_f64 = line_text.contains("f64");
    match (line_f32, line_f64)
    {
        (true, false) => F32Verdict::Yes,
        (false, true) => F32Verdict::No,
        (true, true) => F32Verdict::Ambiguous,
        (false, false) => F32Verdict::No,
    }
}

// =========================================================================
// Parcours du système de fichiers
// =========================================================================

/// Parcourt récursivement `dir`, collecte les fichiers `.rs`, saute les
/// répertoires `SKIP_DIRS`. Les erreurs de lecture d'un sous-répertoire sont
/// signalées sur stderr et n'interrompent pas le parcours.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir)
    {
        Ok(e) => e,
        Err(e) =>
        {
            eprintln!(
                "epsilon-audit: avertissement : lecture de {} impossible : {e}",
                dir.display()
            );
            return;
        },
    };
    for entry in entries
    {
        let entry = match entry
        {
            Ok(e) => e,
            Err(e) =>
            {
                eprintln!(
                    "epsilon-audit: avertissement : entrée illisible dans {} : {e}",
                    dir.display()
                );
                continue;
            },
        };
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir()
        {
            if SKIP_DIRS.contains(&name.as_ref())
            {
                continue;
            }
            collect_rs_files(&path, out);
        }
        else if path.extension().is_some_and(|e| e == "rs")
        {
            out.push(path);
        }
    }
}

// =========================================================================
// Analyse d'un fichier
// =========================================================================

fn analyze_file(root: &Path, path: &Path, findings: &mut Vec<Finding>) {
    let src = match fs::read_to_string(path)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!(
                "epsilon-audit: avertissement : {} illisible ({e}), ignoré",
                path.display()
            );
            return;
        },
    };
    let rel = path.strip_prefix(root).unwrap_or(path).to_path_buf();
    let in_test_dir = path_is_test(&rel);
    let lines: Vec<&str> = src.lines().collect();
    let test_flags = test_active_flags(&lines);

    for lit in scan_float_literals(&src)
    {
        let idx = lit.line.saturating_sub(1);
        let line_text = lines.get(idx).copied().unwrap_or("");
        let test_active = test_flags.get(idx).copied().unwrap_or(false);
        let category = classify(in_test_dir, line_text, test_active);
        let verdict = f32_verdict(&lit, line_text);
        findings.push(Finding {
            file: rel.clone(),
            line: lit.line,
            text: lit.text,
            value: lit.value,
            category,
            f32_verdict: verdict,
            extract: line_text.trim().to_string(),
        });
    }
}

// =========================================================================
// Gate --check : voie sanitized de scirust-gpu/src
// =========================================================================

struct CheckResult {
    violations: Vec<usize>,
    warnings: Vec<usize>,
}

fn under_gpu_src(rel: &Path) -> bool {
    let s = rel.to_string_lossy();
    s.starts_with("scirust-gpu/src/") || s.contains("/scirust-gpu/src/")
}

fn run_check(findings: &[Finding]) -> CheckResult {
    let sigma = f64::from(SIGMA_SANITIZED_F32);
    let mut violations = Vec::new();
    let mut warnings = Vec::new();
    for (i, f) in findings.iter().enumerate()
    {
        if f.category == Category::Test
        {
            continue;
        }
        if !under_gpu_src(&f.file)
        {
            continue;
        }
        if !(f.value > 0.0 && f.value < sigma)
        {
            continue;
        }
        match f.f32_verdict
        {
            F32Verdict::Yes => violations.push(i),
            F32Verdict::Ambiguous => warnings.push(i),
            F32Verdict::No =>
            {},
        }
    }
    CheckResult {
        violations,
        warnings,
    }
}

// =========================================================================
// Génération du rapport Markdown
// =========================================================================

fn crate_name(rel: &Path) -> String {
    let comps: Vec<String> = rel
        .iter()
        .map(|c| c.to_string_lossy().into_owned())
        .collect();
    match comps.first().map(String::as_str)
    {
        Some("examples") if comps.len() >= 2 => format!("examples/{}", comps[1]),
        Some("scirust-som") if comps.len() >= 3 && comps[1] == "crates" =>
        {
            format!("scirust-som/{}", comps[2])
        },
        Some(first) => first.to_string(),
        None => rel.to_string_lossy().into_owned(),
    }
}

fn build_report(root: &Path, findings: &[Finding], check: &CheckResult) -> String {
    let mut body = String::new();
    let _ = writeln!(body, "# Rapport d'audit epsilon — SciRust");
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "Généré par `epsilon-audit` (crate `scirust-sigma`, std-only, parsing lexical)."
    );
    let _ = writeln!(body, "Racine auditée : `{}`.", root.display());
    let _ = writeln!(
        body,
        "Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit."
    );
    let _ = writeln!(body);

    // 1. Totaux par catégorie.
    let mut per_cat = [0usize; 5];
    for f in findings
    {
        per_cat[f.category.index()] += 1;
    }
    let _ = writeln!(body, "## 1. Totaux par catégorie");
    let _ = writeln!(body);
    let _ = writeln!(body, "| Catégorie | Nombre |");
    let _ = writeln!(body, "|---|---:|");
    for cat in Category::ALL
    {
        let _ = writeln!(body, "| {} | {} |", cat.label(), per_cat[cat.index()]);
    }
    let _ = writeln!(body, "| **Total** | **{}** |", findings.len());
    let _ = writeln!(body);

    // 2. Répartition par crate.
    let mut per_crate: std::collections::BTreeMap<String, [usize; 5]> =
        std::collections::BTreeMap::new();
    for f in findings
    {
        let entry = per_crate.entry(crate_name(&f.file)).or_insert([0; 5]);
        entry[f.category.index()] += 1;
    }
    let mut crate_rows: Vec<(String, [usize; 5], usize)> = per_crate
        .into_iter()
        .map(|(k, v)| {
            let total: usize = v.iter().sum();
            (k, v, total)
        })
        .collect();
    crate_rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let _ = writeln!(body, "## 2. Répartition par crate");
    let _ = writeln!(body);
    let _ = writeln!(body, "| Crate | A | B | C | D | U | Total |");
    let _ = writeln!(body, "|---|---:|---:|---:|---:|---:|---:|");
    for (name, counts, total) in &crate_rows
    {
        let _ = writeln!(
            body,
            "| {} | {} | {} | {} | {} | {} | {} |",
            name, counts[0], counts[1], counts[2], counts[3], counts[4], total
        );
    }
    let _ = writeln!(body);

    // 3. Top-20 des valeurs (par valeur, représentant textuel).
    let mut per_value: std::collections::BTreeMap<u64, (String, usize)> =
        std::collections::BTreeMap::new();
    for f in findings
    {
        let key = f.value.to_bits();
        let slot = per_value.entry(key).or_insert((f.text.clone(), 0));
        slot.1 += 1;
    }
    let mut value_rows: Vec<(f64, String, usize)> = per_value
        .into_iter()
        .map(|(bits, (text, cnt))| (f64::from_bits(bits), text, cnt))
        .collect();
    value_rows.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    });
    let _ = writeln!(body, "## 3. Top-20 des valeurs");
    let _ = writeln!(body);
    let _ = writeln!(body, "| Valeur (représentant) | Occurrences |");
    let _ = writeln!(body, "|---|---:|");
    for (_, text, cnt) in value_rows.iter().take(20)
    {
        let _ = writeln!(body, "| `{text}` | {cnt} |");
    }
    let _ = writeln!(body);

    // 4. Voie sanitized (scirust-gpu/src) — vérification σ.
    let _ = writeln!(
        body,
        "## 4. Voie sanitized (`scirust-gpu/src`) — vérification σ"
    );
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "σ_sanitized = `f32::MIN_POSITIVE` = `{:e}` (bits `0x{:08x}`).",
        SIGMA_SANITIZED_F32,
        SIGMA_SANITIZED_F32.to_bits()
    );
    let _ = writeln!(
        body,
        "Règle du gate : hors catégorie C, aucun littéral f32 `> 0` et `< σ` toléré."
    );
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "- Violations bloquantes : **{}**",
        check.violations.len()
    );
    let _ = writeln!(
        body,
        "- Avertissements (ambiguïté f32/f64, non bloquant) : **{}**",
        check.warnings.len()
    );
    if !check.violations.is_empty()
    {
        let _ = writeln!(body);
        for &idx in &check.violations
        {
            if let Some(f) = findings.get(idx)
            {
                let _ = writeln!(
                    body,
                    "  - `{}:{}`  `{}`  {}",
                    f.file.display(),
                    f.line,
                    f.text,
                    f.extract
                );
            }
        }
    }
    if !check.warnings.is_empty()
    {
        let _ = writeln!(body);
        let _ = writeln!(body, "Avertissements :");
        for &idx in &check.warnings
        {
            if let Some(f) = findings.get(idx)
            {
                let _ = writeln!(
                    body,
                    "  - `{}:{}`  `{}`  {}",
                    f.file.display(),
                    f.line,
                    f.text,
                    f.extract
                );
            }
        }
    }
    let _ = writeln!(body);

    // 5. Portée sécurité / sûreté.
    let _ = writeln!(body, "## 5. Portée sécurité / sûreté");
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "L'invariant σ et ce gate d'audit apportent des garanties directement \
         sécuritaires pour une plateforme dont l'argument est le déterminisme \
         bit-à-bit certifiable (verticales sûreté fonctionnelle, OT, navigation) :"
    );
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "- **Neutralise une classe de gardes mortes.** Une garde anti-zéro placée \
         sous σ (dénominateur, `.max`, seuil) est écrasée par `sanitize_f32` sur la \
         voie 3 GPU : le zéro qu'elle prétendait interdire repasse, produisant `Inf`/`NaN` \
         qui se propagent silencieusement. C'est un défaut de sûreté latent, invisible \
         en revue humaine — le gate le rend détectable mécaniquement."
    );
    let _ = writeln!(
        body,
        "- **Contrôle préventif en CI.** `--check` bloque tout *nouveau* littéral f32 \
         sous σ_sanitized introduit dans `scirust-gpu/src` : prévention de régression, \
         pas correction a posteriori. Zéro faux positif bloquant (ambiguïté → WARNING)."
    );
    let _ = writeln!(
        body,
        "- **Aucune surface d'approvisionnement ajoutée.** L'outil est std-only ; `sha2` \
         (déjà au lockfile) n'est utilisé que pour sceller le rapport. Aucun `regex`/`syn`/\
         `serde`, aucun nouveau crate — `deny.toml` et le lockfile restent intacts."
    );
    let _ = writeln!(
        body,
        "- **Intégrité de l'artefact.** Le rapport est scellé par un SHA-256 de son corps \
         (ci-dessous) : toute altération de l'audit est détectable, dans l'esprit du \
         journal d'attestation hash-chaîné de la plateforme."
    );
    let _ = writeln!(
        body,
        "- **Lecture seule.** Le binaire ne modifie, ne supprime, n'écrit aucun fichier \
         source ; seul `--out` écrit le rapport à l'emplacement demandé."
    );
    let _ = writeln!(body);

    // Annexe A — liste complète.
    let mut sorted: Vec<&Finding> = findings.iter().collect();
    sorted.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.line.cmp(&b.line)));
    let _ = writeln!(
        body,
        "## Annexe A — liste complète ({} littéraux)",
        findings.len()
    );
    let _ = writeln!(body);
    let _ = writeln!(body, "```text");
    let _ = writeln!(body, "fichier:ligne  valeur  cat  extrait");
    for f in &sorted
    {
        let extract = if f.extract.chars().count() > 100
        {
            let cut = f
                .extract
                .char_indices()
                .nth(100)
                .map(|(b, _)| b)
                .unwrap_or(f.extract.len());
            format!("{}…", &f.extract[..cut])
        }
        else
        {
            f.extract.clone()
        };
        let _ = writeln!(
            body,
            "{}:{}  {}  {}  {}",
            f.file.display(),
            f.line,
            f.text,
            f.category.code(),
            extract
        );
    }
    let _ = writeln!(body, "```");

    body
}

/// Hachage SHA-256 hexadécimal du corps du rapport (drapeau `report-hash`).
#[cfg(feature = "report-hash")]
fn report_hash(body: &str) -> Option<String> {
    use sha2::{Digest, Sha256};
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest
    {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    Some(s)
}

#[cfg(not(feature = "report-hash"))]
fn report_hash(_body: &str) -> Option<String> {
    None
}

// =========================================================================
// Mode --mine : rapport Markdown du minage d'un dépôt externe
// =========================================================================

fn build_mine_report(root: &Path, outcome: &mine::MineOutcome) -> String {
    let mut body = String::new();
    let _ = writeln!(body, "# Rapport de minage « dead guards »");
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical)."
    );
    let _ = writeln!(body, "Racine minée : `{}`.", root.display());
    let _ = writeln!(
        body,
        "Seuils : M1 (flush FTZ/DAZ) `< {:e}` = `f32::MIN_POSITIVE` ; M2 (inversion) `< {:e}` = `1/f32::MAX`.",
        mine::M1_FLUSH_THRESHOLD,
        mine::M2_INVERSION_THRESHOLD
    );
    let _ = writeln!(
        body,
        "Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit."
    );
    let _ = writeln!(body);

    // 1. Statistiques.
    let m2_count = outcome
        .candidates
        .iter()
        .filter(|c| c.mechanism == mine::Mechanism::M2Inversion)
        .count();
    let _ = writeln!(body, "## 1. Statistiques");
    let _ = writeln!(body);
    let _ = writeln!(body, "| Mesure | Valeur |");
    let _ = writeln!(body, "|---|---:|");
    let _ = writeln!(body, "| Fichiers scannés | {} |", outcome.files_scanned);
    let _ = writeln!(body, "| Lignes scannées | {} |", outcome.lines_scanned);
    let _ = writeln!(
        body,
        "| Fichiers exclus (tests, vendor…) | {} |",
        outcome.files_excluded
    );
    let _ = writeln!(
        body,
        "| Candidats (CONFIRMED-F32 + PROBABLE-F32) | {} |",
        outcome.candidates.len()
    );
    let _ = writeln!(body, "| — dont mécanisme M2 (inversion) | {m2_count} |");
    let _ = writeln!(
        body,
        "| Littéraux sous-seuil UNCERTAIN (non comptés) | {} |",
        outcome.uncertain.len()
    );
    let _ = writeln!(
        body,
        "| Littéraux sous-seuil NOT-F32 (écartés) | {} |",
        outcome.not_f32_count
    );
    let _ = writeln!(
        body,
        "| Drapeaux fast-math/FTZ (fichiers de build) | {} |",
        outcome.fastmath.len()
    );
    let _ = writeln!(body);

    // 2. Candidats.
    let _ = writeln!(body, "## 2. Candidats (à revue manuelle)");
    let _ = writeln!(body);
    if outcome.candidates.is_empty()
    {
        let _ = writeln!(body, "Aucun candidat.");
    }
    else
    {
        let _ = writeln!(
            body,
            "| Fichier:ligne | Langage | Littéral | Mécanisme | Typage | Garde | Extrait |"
        );
        let _ = writeln!(body, "|---|---|---|---|---|---|---|");
        for c in &outcome.candidates
        {
            let _ = writeln!(
                body,
                "| `{}:{}` | {} | `{}` | {} | {} | {} | `{}` |",
                c.file.display(),
                c.line,
                c.language.label(),
                c.literal,
                c.mechanism.label(),
                c.verdict.label(),
                c.guard_marker.unwrap_or("—"),
                c.extract.replace('|', "\\|")
            );
        }
    }
    let _ = writeln!(body);

    // 3. Bloc TSV (agrégation machine).
    let _ = writeln!(body, "## 3. TSV (agrégation)");
    let _ = writeln!(body);
    let _ = writeln!(body, "```tsv");
    let _ = writeln!(
        body,
        "file\tline\tlang\tliteral\tvalue\tmechanism\tverdict\tguard\textract"
    );
    for c in &outcome.candidates
    {
        let _ = writeln!(
            body,
            "{}\t{}\t{}\t{}\t{:e}\t{}\t{}\t{}\t{}",
            c.file.display(),
            c.line,
            c.language.label(),
            c.literal,
            c.value,
            c.mechanism.label(),
            c.verdict.label(),
            c.guard_marker.unwrap_or("-"),
            c.extract
        );
    }
    let _ = writeln!(body, "```");
    let _ = writeln!(body);

    // 4. Drapeaux fast-math (colonne « FTZ probable » de l'étude).
    let _ = writeln!(body, "## 4. Drapeaux fast-math / FTZ");
    let _ = writeln!(body);
    if outcome.fastmath.is_empty()
    {
        let _ = writeln!(body, "Aucun drapeau détecté dans les fichiers de build.");
    }
    else
    {
        // Liste bornée : les gros dépôts CUDA citent ftz des centaines de fois.
        const MAX_LISTED: usize = 40;
        for h in outcome.fastmath.iter().take(MAX_LISTED)
        {
            let _ = writeln!(
                body,
                "- `{}:{}` — `{}`",
                h.file.display(),
                h.line,
                h.pattern
            );
        }
        if outcome.fastmath.len() > MAX_LISTED
        {
            let _ = writeln!(
                body,
                "- … et {} autres occurrences (comptées, non listées).",
                outcome.fastmath.len() - MAX_LISTED
            );
        }
    }
    let _ = writeln!(body);

    // 5. Incertains (transparence, jamais comptés).
    let _ = writeln!(
        body,
        "## 5. Littéraux UNCERTAIN ({} — non comptés comme findings)",
        outcome.uncertain.len()
    );
    let _ = writeln!(body);
    let _ = writeln!(body, "```text");
    for c in &outcome.uncertain
    {
        let _ = writeln!(
            body,
            "{}:{}  {}  {}",
            c.file.display(),
            c.line,
            c.literal,
            c.extract
        );
    }
    let _ = writeln!(body, "```");

    body
}

// =========================================================================
// Arguments
// =========================================================================

struct Args {
    root: PathBuf,
    out: Option<PathBuf>,
    check: bool,
    mine: Option<PathBuf>,
}

fn parse_args() -> Result<Args, AuditError> {
    let mut root = PathBuf::from(".");
    let mut out = None;
    let mut check = false;
    let mut mine = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next()
    {
        match arg.as_str()
        {
            "--check" => check = true,
            "--root" =>
            {
                let v = it
                    .next()
                    .ok_or_else(|| AuditError::Usage("--root attend un chemin".to_string()))?;
                root = PathBuf::from(v);
            },
            "--out" =>
            {
                let v = it
                    .next()
                    .ok_or_else(|| AuditError::Usage("--out attend un chemin".to_string()))?;
                out = Some(PathBuf::from(v));
            },
            "--mine" =>
            {
                let v = it
                    .next()
                    .ok_or_else(|| AuditError::Usage("--mine attend un chemin".to_string()))?;
                mine = Some(PathBuf::from(v));
            },
            "--help" | "-h" =>
            {
                return Err(AuditError::Usage(usage()));
            },
            other if other.starts_with("--root=") =>
            {
                root = PathBuf::from(&other["--root=".len()..]);
            },
            other if other.starts_with("--out=") =>
            {
                out = Some(PathBuf::from(&other["--out=".len()..]));
            },
            other if other.starts_with("--mine=") =>
            {
                mine = Some(PathBuf::from(&other["--mine=".len()..]));
            },
            other =>
            {
                return Err(AuditError::Usage(format!(
                    "argument inconnu : {other}\n{}",
                    usage()
                )));
            },
        }
    }
    Ok(Args {
        root,
        out,
        check,
        mine,
    })
}

fn usage() -> String {
    "usage: epsilon-audit [--root <path>] [--out <file>] [--check] [--mine <dir>]\n\
     \n\
       --root <path>  racine du parcours (défaut : .)\n\
       --out <file>   écrit le rapport Markdown dans <file> (défaut : stdout)\n\
       --check        gate CI : sort ≠ 0 si une garde f32 sous σ_sanitized\n\
                      subsiste hors test dans scirust-gpu/src\n\
       --mine <dir>   mine un dépôt externe (multi-langage) à la recherche de\n\
                      gardes epsilon mortes (M1 flush / M2 inversion) ;\n\
                      rapport Markdown+TSV, code de sortie toujours 0"
        .to_string()
}

// =========================================================================
// Point d'entrée
// =========================================================================

fn run() -> Result<u8, AuditError> {
    let args = parse_args()?;

    // Mode --mine : campagne externe, indépendante de --root/--check.
    if let Some(dir) = &args.mine
    {
        if !dir.is_dir()
        {
            return Err(AuditError::Usage(format!(
                "--mine : répertoire introuvable : {}",
                dir.display()
            )));
        }
        let outcome = mine::mine_dir(dir);
        let body = build_mine_report(dir, &outcome);
        let mut report = body.clone();
        match report_hash(&body)
        {
            Some(hash) =>
            {
                let _ = writeln!(report, "\n---\n\nReport-SHA256: `{hash}`");
            },
            None =>
            {
                let _ = writeln!(
                    report,
                    "\n---\n\nReport-SHA256: (omis — feature `report-hash` désactivée)"
                );
            },
        }
        match &args.out
        {
            Some(path) =>
            {
                fs::write(path, report.as_bytes())?;
                eprintln!(
                    "epsilon-audit --mine: {} fichiers, {} lignes, {} candidats → {}",
                    outcome.files_scanned,
                    outcome.lines_scanned,
                    outcome.candidates.len(),
                    path.display()
                );
            },
            None =>
            {
                let stdout = io::stdout();
                let mut lock = stdout.lock();
                lock.write_all(report.as_bytes())?;
            },
        }
        return Ok(EXIT_OK);
    }

    if !args.root.exists()
    {
        return Err(AuditError::Usage(format!(
            "racine introuvable : {}",
            args.root.display()
        )));
    }
    if !args.root.is_dir()
    {
        return Err(AuditError::Usage(format!(
            "la racine n'est pas un répertoire : {}",
            args.root.display()
        )));
    }

    let mut files = Vec::new();
    collect_rs_files(&args.root, &mut files);
    files.sort();

    let mut findings = Vec::new();
    for path in &files
    {
        analyze_file(&args.root, path, &mut findings);
    }

    let check = run_check(&findings);

    if args.check
    {
        let n_gpu = findings.iter().filter(|f| under_gpu_src(&f.file)).count();
        eprintln!(
            "epsilon-audit --check : {} fichiers, {} littéraux (dont {} sous scirust-gpu/src)",
            files.len(),
            findings.len(),
            n_gpu
        );
        for &idx in &check.warnings
        {
            if let Some(f) = findings.get(idx)
            {
                eprintln!(
                    "epsilon-audit --check: WARNING (ambigu f32/f64) {}:{}  {}",
                    f.file.display(),
                    f.line,
                    f.text
                );
            }
        }
        if check.violations.is_empty()
        {
            eprintln!(
                "epsilon-audit --check: OK — aucune garde f32 sous σ_sanitized ({:e}) dans scirust-gpu/src",
                SIGMA_SANITIZED_F32
            );
            return Ok(EXIT_OK);
        }
        eprintln!(
            "epsilon-audit --check: ÉCHEC — {} garde(s) f32 sous σ_sanitized :",
            check.violations.len()
        );
        for &idx in &check.violations
        {
            if let Some(f) = findings.get(idx)
            {
                eprintln!(
                    "  {}:{}  {}  {}",
                    f.file.display(),
                    f.line,
                    f.text,
                    f.extract
                );
            }
        }
        return Ok(EXIT_CHECK_FAILED);
    }

    // Mode rapport.
    let body = build_report(&args.root, &findings, &check);
    let mut report = body.clone();
    match report_hash(&body)
    {
        Some(hash) =>
        {
            let _ = writeln!(report, "\n---\n\nReport-SHA256: `{hash}`");
        },
        None =>
        {
            let _ = writeln!(
                report,
                "\n---\n\nReport-SHA256: (omis — feature `report-hash` désactivée)"
            );
        },
    }

    match &args.out
    {
        Some(path) =>
        {
            fs::write(path, report.as_bytes())?;
            eprintln!(
                "epsilon-audit: rapport écrit dans {} ({} littéraux)",
                path.display(),
                findings.len()
            );
        },
        None =>
        {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(report.as_bytes())?;
        },
    }

    Ok(EXIT_OK)
}

fn main() -> ExitCode {
    match run()
    {
        Ok(code) => ExitCode::from(code),
        Err(AuditError::Usage(m)) =>
        {
            eprintln!("epsilon-audit: {m}");
            ExitCode::from(EXIT_USAGE)
        },
        Err(AuditError::Io(e)) =>
        {
            eprintln!("epsilon-audit: erreur d'E/S : {e}");
            ExitCode::from(EXIT_IO)
        },
    }
}
