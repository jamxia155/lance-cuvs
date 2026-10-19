// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

fn main() {
    if let Err(error) = lance_cuvs_backend_cu12_local::cagra_bench::run() {
        eprintln!("cagra_assign_bench failed: {error}");
        std::process::exit(1);
    }
}
