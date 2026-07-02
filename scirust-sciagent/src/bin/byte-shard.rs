use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "byte-shard",
    about = "Quickly pack .rs files into raw byte shards"
)]
struct Args {
    #[arg(short, long)]
    input: Vec<String>,

    #[arg(short, long, default_value = "./shards")]
    output: PathBuf,

    #[arg(long, default_value = "rs")]
    extension: String,

    #[arg(long)]
    recursive: bool,

    #[arg(long, default_value_t = 1_000_000)]
    tokens_per_shard: usize,

    #[arg(long, default_value_t = 0)]
    max_files: usize,
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.output).expect("Cannot create output dir");

    let mut all_tokens: Vec<u8> = Vec::new();
    let mut file_count = 0;

    for path in &args.input
    {
        let p = Path::new(path);
        if p.is_file()
        {
            if let Ok(content) = fs::read(p)
            {
                all_tokens.extend(&content);
                // Add BOS/EOS markers (0x00 = <bos>, 0xFF = <eos>)
                all_tokens.push(0xFF);
                file_count += 1;
                if args.max_files > 0 && file_count >= args.max_files
                {
                    break;
                }
            }
        }
        else if p.is_dir() && args.recursive
        {
            file_count += collect_dir(p, &args.extension, &mut all_tokens, args.max_files);
            if args.max_files > 0 && file_count >= args.max_files
            {
                break;
            }
        }
    }

    eprintln!("Total files: {file_count}, tokens: {}", all_tokens.len());
    eprintln!("Packing into shards of {} tokens...", args.tokens_per_shard);

    let shard_size = args.tokens_per_shard;
    let num_shards = all_tokens.len().div_ceil(shard_size);

    for shard_idx in 0..num_shards
    {
        let start = shard_idx * shard_size;
        let end = std::cmp::min(start + shard_size, all_tokens.len());
        let shard_data = &all_tokens[start..end];

        let shard_path = args.output.join(format!("shard_{:04}.bin", shard_idx));
        let mut f = fs::File::create(&shard_path).expect("Cannot create shard file");
        f.write_all(shard_data).expect("Write error");

        eprintln!(
            "Shard {:04}: {} tokens -> {:?}",
            shard_idx,
            shard_data.len(),
            shard_path
        );
    }

    eprintln!("Done: {num_shards} shards written to {:?}", args.output);
}

fn collect_dir(dir: &Path, ext: &str, tokens: &mut Vec<u8>, max_files: usize) -> usize {
    let mut count = 0;
    if let Ok(entries) = fs::read_dir(dir)
    {
        for entry in entries.flatten()
        {
            if max_files > 0 && count >= max_files
            {
                return count;
            }
            let path = entry.path();
            if path.is_dir()
            {
                count += collect_dir(&path, ext, tokens, max_files);
            }
            else if path.extension().and_then(|e| e.to_str()) == Some(ext)
            {
                if let Ok(content) = fs::read(&path)
                {
                    tokens.extend(&content);
                    tokens.push(0xFF);
                    count += 1;
                }
            }
        }
    }
    count
}
