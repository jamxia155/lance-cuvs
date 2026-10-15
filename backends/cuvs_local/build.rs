// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

fn main() {
    // cuvs-sys (`links = "cuvs"`) reports the directory of the libcuvs_c it
    // found. Embed it as an rpath so the extension loads that local build
    // without LD_LIBRARY_PATH or LD_PRELOAD; libcuvs_c's own rpath then finds
    // its librmm/libraft/... from the conda env it was built in.
    println!("cargo::rerun-if-env-changed=DEP_CUVS_LIB");
    if let Ok(lib_dir) = std::env::var("DEP_CUVS_LIB") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
