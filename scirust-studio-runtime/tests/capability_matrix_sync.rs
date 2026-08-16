use scirust_studio_runtime::all_adapters;

#[test]
fn current_desktop_capability_count_matches_runtime_registry() {
    let matrix = include_str!("../../docs/studio/CAPABILITY_MATRIX.md");
    let current_section = matrix
        .split("## The rest of the workspace")
        .next()
        .expect("capability matrix must contain its current-state section");
    let expected = format!("**Desktop-exposed: {} capabilities", all_adapters().len());

    assert!(
        current_section.contains(&expected),
        "current Studio capability matrix is out of sync with the runtime registry: expected `{expected}`"
    );
}
