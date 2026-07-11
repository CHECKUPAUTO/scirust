// examples/obd2_diagnostic/src/main.rs
//
// SciRust — Assistant de diagnostic automobile OBD2
// ==================================================
//
// Une PETITE IA (un réseau de neurones MLP) qui :
//   1. lit un code défaut OBD2 (P0171, P0300, P0420, ...) + quelques symptômes,
//   2. prédit la CAUSE RACINE la plus probable,
//   3. classe TOUTES les causes possibles par probabilité (assistance au
//      diagnostic — comme un mécanicien qui donne ses hypothèses par ordre),
//   4. propose l'action de réparation / le prochain test à faire.
//
// Tout est écrit avec le coeur de SciRust (`scirust-core`) : le même moteur
// d'autograd (Tape), les mêmes couches (Linear/ReLU) et le même optimiseur
// (Adam) que la démo `quickstart_v2`. Ici on l'a juste SPÉCIALISÉ pour un
// métier : le diagnostic automobile.
//
// Déterminisme : même graine (seed) => mêmes nombres à chaque exécution.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

// --------------------------------------------------------------------------
// 1. LE PROBLÈME, EN CHIFFRES
// --------------------------------------------------------------------------
//
// Une IA ne comprend pas les mots : elle ne manipule que des nombres. On doit
// donc transformer une "situation atelier" en un vecteur de 7 nombres.
//
// LES 7 ENTRÉES (features) qui décrivent un cas :
//   [0] code "mélange trop pauvre"   (P0171/P0174)      -> 1 si présent, sinon 0
//   [1] code "ratés d'allumage"      (P0300..P0308)     -> 1 / 0
//   [2] code "catalyseur"            (P0420/P0430)      -> 1 / 0
//   [3] code "fuite EVAP"            (P0442/P0455/...)  -> 1 / 0
//   [4] correction carburant long terme (LTFT), normalisée dans [0..1]
//         0.5 = 0 %   (normal)   ;   1.0 = +25 % (le moteur rajoute beaucoup
//         de carburant => il compense une entrée d'air / un manque)
//   [5] débit d'air mesuré (MAF) anormalement bas       -> 1 / 0
//   [6] ralenti instable / vibrations signalées         -> 1 / 0
const N_FEATURES: usize = 7;

// LES 5 CAUSES RACINES que l'IA doit savoir distinguer (les "classes") :
const N_CLASSES: usize = 5;
const CAUSES: [&str; N_CLASSES] = [
    "Prise d'air / fuite de depression (durite, joint d'admission)",
    "Capteur de debit d'air (MAF) encrasse ou defectueux",
    "Systeme d'allumage (bougies / bobines)",
    "Convertisseur catalytique en fin de vie",
    "Fuite du circuit EVAP (bouchon de reservoir, durite)",
];

// L'ASSISTANCE : pour chaque cause, l'action concrète recommandée.
const ACTIONS: [&str; N_CLASSES] = [
    "Tester au nettoyant frein / fumigene autour de l'admission ; verifier durites de depression et joint de collecteur.",
    "Nettoyer le capteur MAF (nettoyant specifique), comparer g/s au ralenti a la valeur constructeur ; remplacer si hors plage.",
    "Controler bougies et bobines cylindre par cylindre ; permuter la bobine du cylindre en defaut pour voir si le rate suit.",
    "Comparer sonde O2 amont/aval ; verifier l'absence de rate moteur (qui detruit le catalyseur) AVANT de remplacer le catalyseur.",
    "Reserrer / remplacer le bouchon de reservoir, puis test d'etancheite (smoke test) du circuit EVAP.",
];

// --------------------------------------------------------------------------
// 2. LES DONNÉES D'ENTRAÎNEMENT (le "vécu atelier" dont l'IA apprend)
// --------------------------------------------------------------------------
//
// Chaque ligne = (7 features, cause racine connue). Dans la vraie vie ces
// exemples viendraient d'un historique de réparations validées.
//
// Astuce pédagogique : les causes 0 (prise d'air) et 1 (MAF) donnent le MÊME
// code (mélange pauvre) et la MÊME correction carburant élevée. La SEULE chose
// qui les départage est la feature [5] (débit d'air bas). C'est exactement le
// genre de "désambiguïsation de cause racine" qu'un bon diagnostic exige, et
// que l'IA doit apprendre à repérer.
#[rustfmt::skip]
fn training_data() -> Vec<([f32; N_FEATURES], usize)> {
    vec![
        // --- Cause 0 : prise d'air (pauvre + trim haut + MAF NORMAL + ralenti instable) ---
        ([1.0, 0.0, 0.0, 0.0, 0.85, 0.0, 1.0], 0),
        ([1.0, 0.0, 0.0, 0.0, 0.78, 0.0, 1.0], 0),
        ([1.0, 0.0, 0.0, 0.0, 0.90, 0.0, 1.0], 0),
        ([1.0, 0.0, 0.0, 0.0, 0.72, 0.0, 0.0], 0),
        ([1.0, 0.0, 0.0, 0.0, 0.82, 0.0, 1.0], 0),

        // --- Cause 1 : capteur MAF (pauvre + trim haut + MAF BAS) ---
        ([1.0, 0.0, 0.0, 0.0, 0.80, 1.0, 0.0], 1),
        ([1.0, 0.0, 0.0, 0.0, 0.75, 1.0, 1.0], 1),
        ([1.0, 0.0, 0.0, 0.0, 0.88, 1.0, 0.0], 1),
        ([1.0, 0.0, 0.0, 0.0, 0.70, 1.0, 0.0], 1),
        ([1.0, 0.0, 0.0, 0.0, 0.83, 1.0, 1.0], 1),

        // --- Cause 2 : allumage (rate d'allumage + trim NORMAL + ralenti instable) ---
        ([0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 1.0], 2),
        ([0.0, 1.0, 0.0, 0.0, 0.55, 0.0, 1.0], 2),
        ([0.0, 1.0, 0.0, 0.0, 0.45, 0.0, 1.0], 2),
        ([0.0, 1.0, 0.0, 0.0, 0.52, 0.0, 0.0], 2),
        ([0.0, 1.0, 0.0, 0.0, 0.48, 0.0, 1.0], 2),

        // --- Cause 3 : catalyseur (code cata + tout le reste calme) ---
        ([0.0, 0.0, 1.0, 0.0, 0.50, 0.0, 0.0], 3),
        ([0.0, 0.0, 1.0, 0.0, 0.53, 0.0, 0.0], 3),
        ([0.0, 0.0, 1.0, 0.0, 0.47, 0.0, 0.0], 3),
        ([0.0, 0.0, 1.0, 0.0, 0.51, 0.0, 0.0], 3),
        ([0.0, 0.0, 1.0, 0.0, 0.49, 0.0, 0.0], 3),

        // --- Cause 4 : fuite EVAP (code EVAP + tout le reste calme) ---
        ([0.0, 0.0, 0.0, 1.0, 0.50, 0.0, 0.0], 4),
        ([0.0, 0.0, 0.0, 1.0, 0.52, 0.0, 0.0], 4),
        ([0.0, 0.0, 0.0, 1.0, 0.48, 0.0, 0.0], 4),
        ([0.0, 0.0, 0.0, 1.0, 0.51, 0.0, 0.0], 4),
        ([0.0, 0.0, 0.0, 1.0, 0.49, 0.0, 0.0], 4),
    ]
}

// --------------------------------------------------------------------------
// 3. OUTILS D'INFÉRENCE : softmax (logits -> probabilités qui somment à 100 %)
// --------------------------------------------------------------------------
fn softmax(logits: &[f32]) -> Vec<f32> {
    // "max-trick" : on retire le max avant l'exponentielle pour rester stable.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// --------------------------------------------------------------------------
// 4. PROGRAMME PRINCIPAL
// --------------------------------------------------------------------------
fn main() {
    println!("=== SciRust — Assistant de diagnostic automobile OBD2 ===\n");

    let data = training_data();

    // ---- Le modèle : MLP 7 -> 16 -> 5 ----
    // 7 entrées (le cas), une couche cachée de 16 neurones + ReLU pour la
    // non-linéarité, puis 5 sorties (une par cause racine).
    let mut rng = PcgEngine::new(42); // graine fixe => résultat reproductible
    let mut model = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            16,
            &KaimingNormal,
            &Zeros,
            &mut rng,
        ))
        .add(ReLU::new())
        .add(Linear::new(16, N_CLASSES, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.05);

    let n_epochs = 400;
    println!(
        "Entrainement : {} cas d'atelier, {} epochs, Adam(lr=0.05)\n",
        data.len(),
        n_epochs
    );

    // ---- Boucle d'entraînement ----
    for epoch in 0..n_epochs
    {
        let mut epoch_loss = 0.0;

        for (features, label) in &data
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));

            // cible "one-hot" : un 1.0 sur la bonne cause, 0.0 ailleurs
            let mut target_data = vec![0.0; N_CLASSES];
            target_data[*label] = 1.0;
            let target = tape.input(Tensor::from_vec(target_data, 1, N_CLASSES));

            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
        }

        if epoch == 0 || (epoch + 1) % 100 == 0
        {
            println!(
                "  Epoch {:>4} : loss = {:.6}",
                epoch + 1,
                epoch_loss / data.len() as f32
            );
        }
    }

    // ---- Vérification : l'IA a-t-elle bien appris ses cas ? ----
    let mut correct = 0;
    for (features, label) in &data
    {
        if predict_class(&mut model, features) == *label
        {
            correct += 1;
        }
    }
    println!(
        "\nPrecision sur les cas connus : {}/{}\n",
        correct,
        data.len()
    );

    // ---- Inférence sur des cas NOUVEAUX (jamais vus à l'entraînement) ----
    println!("======================================================");
    println!("   DIAGNOSTIC DE CAS REELS (nouveaux pour l'IA)");
    println!("======================================================");

    // (code affiché, description humaine, vecteur de 7 features)
    let cas_reels: [(&str, &str, [f32; N_FEATURES]); 5] = [
        (
            "P0171",
            "Melange pauvre, correction carburant +21%, debit d'air NORMAL, ralenti instable",
            [1.0, 0.0, 0.0, 0.0, 0.92, 0.0, 1.0],
        ),
        (
            "P0171",
            "Melange pauvre, correction carburant +18%, debit d'air BAS, ralenti ok",
            [1.0, 0.0, 0.0, 0.0, 0.86, 1.0, 0.0],
        ),
        (
            "P0301",
            "Rate d'allumage cylindre 1, correction carburant normale, ralenti tremblant",
            [0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 1.0],
        ),
        (
            "P0420",
            "Rendement catalyseur sous le seuil, sonde aval qui suit l'amont",
            [0.0, 0.0, 1.0, 0.0, 0.51, 0.0, 0.0],
        ),
        (
            "P0455",
            "Grosse fuite EVAP detectee, aucun autre symptome",
            [0.0, 0.0, 0.0, 1.0, 0.50, 0.0, 0.0],
        ),
    ];

    for (code, description, features) in &cas_reels
    {
        diagnose(&mut model, code, description, features);
    }
}

/// Renvoie l'indice de la cause la plus probable (argmax des logits).
fn predict_class(model: &mut Sequential, features: &[f32; N_FEATURES]) -> usize {
    let tape = Tape::new();
    let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));
    let logits = model.forward(&tape, x);
    let scores = tape.value(logits.idx());
    let mut best = 0;
    for i in 1..N_CLASSES
    {
        if scores.data[i] > scores.data[best]
        {
            best = i;
        }
    }
    best
}

/// Diagnostic complet et lisible d'un cas : hypothèses classées + action.
fn diagnose(model: &mut Sequential, code: &str, description: &str, features: &[f32; N_FEATURES]) {
    let tape = Tape::new();
    let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));
    let logits = model.forward(&tape, x);
    let probs = softmax(&tape.value(logits.idx()).data);

    // Trier les causes par probabilité décroissante.
    let mut ranked: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n----------------------------------------------------");
    println!("  Code       : {}", code);
    println!("  Symptomes  : {}", description);
    println!("  Hypotheses (classees par l'IA) :");
    for (rank, (cause_idx, p)) in ranked.iter().enumerate()
    {
        let marker = if rank == 0 { ">>" } else { "  " };
        println!("    {} {:>5.1}%  {}", marker, p * 100.0, CAUSES[*cause_idx]);
    }

    let (top_cause, top_p) = ranked[0];
    println!(
        "\n  => CAUSE RACINE LA PLUS PROBABLE ({:.1}%) :\n     {}",
        top_p * 100.0,
        CAUSES[top_cause]
    );
    println!("  => ACTION RECOMMANDEE :\n     {}", ACTIONS[top_cause]);
}
