//! Thin dispatcher into `scirust-trader`'s CLI.

pub fn run(args: &[String]) -> u8 {
    scirust_trader::cli::run(args)
}