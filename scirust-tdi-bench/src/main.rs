use scirust_tdi::{
    Action, State, TableSystem, TdiSignature, analyze_recovery, explore,
    uniform_future_block_entropy_bits,
};

fn cycle_four() -> Result<TableSystem, String> {
    let mut system =
        TableSystem::new(2).map_err(|error| format!("cannot create cycle-4: {error:?}"))?;

    for (source, target) in [(0, 1), (1, 2), (2, 3), (3, 0)]
    {
        system
            .insert(
                State::new(source, 2).map_err(|error| error.to_string())?,
                Action::Noop,
                vec![State::new(target, 2).map_err(|error| error.to_string())?],
            )
            .map_err(|error| format!("cannot insert cycle-4 transition: {error:?}"))?;
    }

    Ok(system)
}

fn two_cycles() -> Result<TableSystem, String> {
    let mut system =
        TableSystem::new(2).map_err(|error| format!("cannot create two-cycles: {error:?}"))?;

    for (source, target) in [(0, 1), (1, 0), (2, 3), (3, 2)]
    {
        system
            .insert(
                State::new(source, 2).map_err(|error| error.to_string())?,
                Action::Noop,
                vec![State::new(target, 2).map_err(|error| error.to_string())?],
            )
            .map_err(|error| format!("cannot insert two-cycles transition: {error:?}"))?;
    }

    Ok(system)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cycle = cycle_four()?;
    let pairs = two_cycles()?;
    let initial = State::new(0, 2)?;
    let actions = [Action::Noop, Action::Noop];
    let perturbation = Action::Flip { node: 1 };

    let cycle_entropy = uniform_future_block_entropy_bits(&cycle, Action::Noop, 8)
        .map_err(|error| format!("cycle entropy failed: {error:?}"))?;

    let pairs_entropy = uniform_future_block_entropy_bits(&pairs, Action::Noop, 8)
        .map_err(|error| format!("pairs entropy failed: {error:?}"))?;

    let cycle_signature = TdiSignature::from_report(
        &explore(&cycle, initial, &actions)
            .map_err(|error| format!("cycle exploration failed: {error:?}"))?,
    )
    .map_err(|error| format!("cycle signature failed: {error:?}"))?;

    let pairs_signature = TdiSignature::from_report(
        &explore(&pairs, initial, &actions)
            .map_err(|error| format!("pairs exploration failed: {error:?}"))?,
    )
    .map_err(|error| format!("pairs signature failed: {error:?}"))?;

    let cycle_recovery = analyze_recovery(&cycle, initial, perturbation, 16)
        .map_err(|error| format!("cycle recovery failed: {error:?}"))?;

    let pairs_recovery = analyze_recovery(&pairs, initial, perturbation, 16)
        .map_err(|error| format!("pairs recovery failed: {error:?}"))?;

    println!("TDI-1 adversarial recovery demonstration");
    println!("cycle-4 block entropy    : {cycle_entropy:.1} bits");
    println!("two-cycles block entropy : {pairs_entropy:.1} bits");
    println!(
        "cycle-4 return profile   : {:?}",
        cycle_signature.return_profile()
    );
    println!(
        "two-cycles return profile: {:?}",
        pairs_signature.return_profile()
    );
    println!("cycle-4 recovered        : {}", cycle_recovery.recovered());
    println!("two-cycles recovered     : {}", pairs_recovery.recovered());

    Ok(())
}
