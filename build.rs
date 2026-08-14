fn main() {
    // R loads the library and supplies its C API symbols at runtime. macOS
    // otherwise requires every symbol to be resolved while linking the dylib.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
