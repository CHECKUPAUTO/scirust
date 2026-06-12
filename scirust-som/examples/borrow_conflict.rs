// A mutable borrow taken while a shared borrow is live (E0502), under the
// oracle's lexical borrow model.
fn conflict(data: Vec<u8>) {
    let shared = &data;
    let exclusive = &mut data;
    let n = shared.len();
    drop(exclusive);
    drop(n);
}
