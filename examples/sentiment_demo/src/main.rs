use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{CrossEntropyLoss, Loss, PcgEngine};
use scirust_learning::nlp::sentiment::{SentimentPipeline, SentimentPolarity};
use scirust_learning::nlp::tokenization::{SimpleTokenizer, Tokenizer};

fn main() {
    println!("=== SciRust Sentiment Analysis MVP Demo (v1.1) ===\n");

    let mut rng = PcgEngine::new(42);

    // 1. Préparation des données synthétiques plus riches (FR/EN)
    let train_data = vec![
        // Positifs
        ("C'est génial", 1),
        ("J'adore ce framework", 1),
        ("C'est fantastique et rapide", 1),
        ("Le déterminisme est parfait", 1),
        ("Excellent travail", 1),
        ("Great job", 1),
        ("I love this", 1),
        ("Amazing performance", 1),
        ("Very good", 1),
        // Négatifs
        ("C'est nul", 0),
        ("Je déteste les bugs", 0),
        ("C'est très lent et mauvais", 0),
        ("Pas bon du tout", 0),
        ("Mauvaise expérience", 0),
        ("It's terrible", 0),
        ("I hate this bug", 0),
        ("Very slow and bad", 0),
        ("Disappointing", 0),
    ];

    // 2. Initialisation du tokeniseur
    let all_texts: Vec<&str> = train_data.iter().map(|(t, _)| *t).collect();
    let tokenizer = SimpleTokenizer::build(&all_texts, 1);
    println!("Taille du vocabulaire : {}", tokenizer.vocab_size());

    // 3. Création du pipeline
    let mut pipeline = SentimentPipeline::new(
        Box::new(tokenizer),
        32, // embed_dim (augmenté pour v1.1)
        10, // max_seq_len
        &mut rng,
    );

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.01);

    // 4. Entraînement
    println!("\nEntraînement en cours...");
    for epoch in 0..100
    {
        let mut total_loss = 0.0;
        for (text, label) in &train_data
        {
            let tape = Tape::new();
            let logits = pipeline.forward(&tape, text);

            let mut target_data = vec![0.0; 2];
            target_data[*label] = 1.0;
            let target = tape.input(Tensor::from_vec(target_data, 1, 2));

            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());

            opt.step(&pipeline.parameter_indices(), &tape);
            pipeline.sync(&tape);

            total_loss += tape.value(loss.idx()).data[0];
        }
        if (epoch + 1) % 20 == 0
        {
            println!(
                "  Époque {}/100, Loss: {:.4}",
                epoch + 1,
                total_loss / train_data.len() as f32
            );
        }
    }

    // 5. Inférence multi-langue
    println!("\n--- Inférence Multi-langue ---");
    let test_sentences = vec![
        "C'est génial et rapide",
        "C'est vraiment mauvais",
        "I love the determinism",
        "Very slow experience",
        "Excellent",
        "Terrible bugs",
    ];

    for text in test_sentences
    {
        let result = pipeline.predict(text);
        let sentiment_str = match result.polarity
        {
            SentimentPolarity::Positive => "POSITIF",
            SentimentPolarity::Negative => "NÉGATIF",
            _ => "NEUTRE",
        };
        println!("Texte: \"{}\"", text);
        println!(
            "  Sentiment: {} (confiance: {:.2}%)",
            sentiment_str,
            result.confidence * 100.0
        );
    }
}
