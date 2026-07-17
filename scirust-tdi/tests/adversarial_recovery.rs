use scirust_tdi::{
    Action, State, TableSystem, analyze_recovery, uniform_future_block_entropy_bits,
};

fn cycle_four() -> TableSystem {
    let mut system = TableSystem::new(2).expect("valid width");

    for (source, target) in [(0, 1), (1, 2), (2, 3), (3, 0)]
    {
        system
            .insert(
                State::new(source, 2).expect("valid source"),
                Action::Noop,
                vec![State::new(target, 2).expect("valid target")],
            )
            .expect("valid transition");
    }

    system
}

fn two_cycles() -> TableSystem {
    let mut system = TableSystem::new(2).expect("valid width");

    for (source, target) in [(0, 1), (1, 0), (2, 3), (3, 2)]
    {
        system
            .insert(
                State::new(source, 2).expect("valid source"),
                Action::Noop,
                vec![State::new(target, 2).expect("valid target")],
            )
            .expect("valid transition");
    }

    system
}

#[test]
fn equal_shannon_entropy_does_not_imply_equal_recovery() {
    let cycle = cycle_four();
    let pairs = two_cycles();
    let initial = State::new(0, 2).expect("valid initial state");
    let perturbation = Action::Flip { node: 1 };

    let cycle_entropy =
        uniform_future_block_entropy_bits(&cycle, Action::Noop, 8).expect("entropy succeeds");

    let pairs_entropy =
        uniform_future_block_entropy_bits(&pairs, Action::Noop, 8).expect("entropy succeeds");

    assert_eq!(cycle_entropy, pairs_entropy);
    assert_eq!(cycle_entropy, 2.0);

    let cycle_recovery =
        analyze_recovery(&cycle, initial, perturbation, 16).expect("cycle recovery succeeds");

    let pairs_recovery =
        analyze_recovery(&pairs, initial, perturbation, 16).expect("pair recovery succeeds");

    assert!(cycle_recovery.recovered());
    assert!(!pairs_recovery.recovered());
}
