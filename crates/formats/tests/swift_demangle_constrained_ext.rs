use craft_anything_formats::swift_demangle::{Status, demangle};

fn dm(symbol: &str) -> String {
    let d = demangle(symbol);
    assert_eq!(d.status, Status::Demangled, "{symbol}: {:?}", d.reason);
    d.demangled.unwrap_or_default()
}

#[test]
fn constrained_extension_r_form_has_no_leading_parameters() {
    assert_eq!(dm("_$sSDsSHR_rlE"), "Swift.Dictionary< where B: Swift.Hashable>");
}

#[test]
fn unconstrained_extension_l_form_lists_introduced_parameters() {
    assert_eq!(dm("_$sSDsSHR_lE"), "Swift.Dictionary<A where B: Swift.Hashable>");
}

#[test]
fn constrained_extension_needs_a_requirement() {
    let d = demangle("_$sSDsSHrlE");
    assert_eq!(d.status, Status::Invalid);
}

#[test]
fn constrained_extension_rejects_second_r_operator() {
    let d = demangle("_$sSDsSHR_rrlE");
    assert_eq!(d.status, Status::Invalid);
}
