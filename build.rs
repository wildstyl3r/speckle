use std::process::Command;

fn main() {
    let hash_output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .expect("Failed to execute git");
    let git_hash = String::from_utf8(hash_output.stdout)
        .unwrap()
        .trim()
        .to_string();

    let status_output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .expect("Failed to execute git status");

    let is_dirty = !status_output.stdout.is_empty();

    let version = if is_dirty {
        format!("{}-dirty", git_hash)
    } else {
        git_hash
    };

    println!("cargo:rustc-env=GIT_HASH={}", version);
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-changed=.git/HEAD");
}
