// Real Rust analyzed by `som-analyze`. `String` is non-Copy, so this is an
// unambiguous use-after-move that rustc rejects with E0382.
fn process(input: String) {
    let owned = input;
    let moved = owned;   // moves `owned`
    let oops = owned;    // E0382: use of moved value `owned`
    drop(oops);
    drop(moved);
}
