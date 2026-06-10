use scirust_symbolic::Expr;
use scirust_symreg::discover;

fn show(name: &str, vars: &[&str], front: &[(usize, f64, Expr)]) {
    println!("### {name}   (variables: {})", vars.join(", "));
    println!("  {:>6} {:>11}   formule", "taille", "mse");
    for (s, m, e) in front
    {
        println!("  {:>6} {:>11.2e}   {}", s, m, e);
    }
    println!();
}

fn main() {
    let mut data = vec![];
    let xs: [f64; 6] = [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0];
    let ys: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for &xx in &xs
    {
        for &yy in &ys
        {
            data.push((vec![xx, yy], xx * yy + xx.sin()));
        }
    }
    show(
        "f(x,y) = x*y + sin(x)",
        &["x", "y"],
        &discover(&data, &["x", "y"], &[1, 2, 3], 200, 22, 35, 25),
    );

    let mut data2 = vec![];
    let mut x: f64 = -4.0;
    while x <= 4.0 + 1e-9
    {
        data2.push((vec![x], x / (1.0 + x * x)));
        x += 0.2;
    }
    show(
        "f(x) = x / (1 + x^2)",
        &["x"],
        &discover(&data2, &["x"], &[1, 2, 3], 200, 22, 35, 25),
    );
}
