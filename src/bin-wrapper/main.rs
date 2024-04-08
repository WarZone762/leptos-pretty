//! Because rustfmt links to the specific toolchain's libraries, we need to locate them with
//! rustup

use std::{env, io::Write, process::Command};

const TOOLCHAIN: &str = "nightly-2023-12-28";

fn main() {
    let toolchains = &Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .expect("failed to start rustup")
        .stdout;
    let toolchains = std::str::from_utf8(toolchains).unwrap();
    if !toolchains.contains(TOOLCHAIN) {
        let mut buf = String::with_capacity(2);
        print!(
            "warning: leptos-pretty requires '{TOOLCHAIN}' toolchain to be installed, install it?(y/N)"
        );
        std::io::stdout().flush().expect("failed to flush stdout");
        std::io::stdin()
            .read_line(&mut buf)
            .expect("failed to read line from stdin");
        if !buf.starts_with(['y', 'Y']) {
            return;
        }

        Command::new("rustup")
            .args([
                "toolchain",
                "install",
                TOOLCHAIN,
                "--profile",
                "minimal",
                "--component",
                "llvm-tools",
                "rustc-dev",
            ])
            .spawn()
            .expect("failed to start rustup")
            .wait()
            .expect("rustup failed");
    }

    let args: Vec<_> = env::args().collect();
    Command::new("rustup")
        .args(["run", TOOLCHAIN, "leptos-pretty-format"])
        .args(&args[1..])
        .status()
        .expect("failed to start leptos-pretty-format");
}
