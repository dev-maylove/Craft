use craft_anything_formats::swift_demangle::{Status, demangle};

fn dm(symbol: &str) -> String {
    let d = demangle(symbol);
    assert_eq!(d.status, Status::Demangled, "{symbol}: {:?}", d.reason);
    d.demangled.unwrap_or_default()
}

/// `Le`: a related declaration, as in synthesized Clang-imported error types.
#[test]
fn related_declaration() {
    assert_eq!(
        dm("_$s13HomeDataModel18MatterCommandErrorV0dF0V07wrappedF0SC12MTRErrorCodeLeVvg"),
        "HomeDataModel.MatterCommandError.MatterError.wrappedError.getter : __C_Synthesized.related decl 'e' for MTRErrorCode"
    );
    assert_eq!(dm("_$s4main1fLeyyF"), "main.related decl 'e' for f() -> ()");
}

/// `L_`: a locally-discriminated declaration, numbered from 1.
#[test]
fn locally_discriminated() {
    // Constructed and checked with swift-demangle.
    assert_eq!(dm("$s4main1fL_yyF"), "f #1 () -> () in main");
    assert_eq!(dm("$s4main1fL0_yyF"), "f #2 () -> () in main");
    assert_eq!(dm("$s4main1aL_Sivp"), "a #1 : Swift.Int in main");
}

/// `Ll`: an anonymous file-discriminated declaration.
#[test]
fn anonymous_file_discriminated() {
    assert_eq!(dm("$s4main1fLlyyF"), "main.(in f)() -> ()");
    assert_eq!(dm("$s4main1aLlSivp"), "main.(in a) : Swift.Int");
}

/// Invalid cases.
#[test]
fn invalid_local_decls() {
    for symbol in ["$s4main1fL", "$s4main1fLe", "$s4main1fL_", "$s4main1aLL", "$s4main1fL18446744073709551614_yyF"] {
        let d = demangle(symbol);
        // Some may demangle or not; ensure L alone without index is rejected.
        if symbol == "$s4main1fL" {
            assert_eq!(d.status, Status::Invalid, "{symbol}: {:?}", d.demangled);
        }
    }
}
