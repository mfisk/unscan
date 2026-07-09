//! Atomic file write helpers.
//!
//! Write to a `.tmp` sibling and rename on completion so readers never
//! see a partially-written file.  The rename is atomic on Linux.

use std::path::{Path, PathBuf};

/// Returns a temporary path by appending `.tmp` to the given path.
pub(crate) fn tmp_for(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".tmp");
    PathBuf::from(s)
}
