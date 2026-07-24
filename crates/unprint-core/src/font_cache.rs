//! LRU font-data cache for on-demand font loading.
//!
//! After the character index is built, font file bytes are dropped from the
//! catalog to free ~1 GB of RAM.  This cache loads font files from disk as
//! needed, keeping the N most recently used in memory.  **All post-index font
//! access should go through this cache.**
//!
//! Thread-safe: the inner state is behind a `Mutex`, suitable for use from
//! Rayon `par_iter` closures.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Default number of fonts to keep in cache.
pub const DEFAULT_CAPACITY: usize = 64;

pub struct FontCache {
    inner: Mutex<LruInner>,
}

struct LruInner {
    entries: HashMap<PathBuf, Arc<Vec<u8>>>,
    order: VecDeque<PathBuf>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl FontCache {
    pub fn new(capacity: usize) -> Self {
        FontCache {
            inner: Mutex::new(LruInner {
                entries: HashMap::with_capacity(capacity),
                order: VecDeque::with_capacity(capacity),
                capacity,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Load font data from cache or disk.  Returns a reference-counted
    /// handle to the raw bytes — cheap to clone, callers borrow `&[u8]`
    /// from the Arc for as long as they need.
    pub fn load(&self, path: &Path) -> io::Result<Arc<Vec<u8>>> {
        let mut inner = self.inner.lock().unwrap();

        if let Some(data) = inner.entries.get(path).cloned() {
            // Promote in LRU order
            if let Some(pos) = inner.order.iter().position(|p| p == path) {
                inner.order.remove(pos);
            }
            inner.order.push_back(path.to_path_buf());
            inner.hits += 1;
            return Ok(data);
        }

        // Not cached — read from disk.  Lock is held, but reads are fast
        // from the OS page cache (fonts were just scanned during index build).
        inner.misses += 1;
        let bytes = std::fs::read(path)?;
        let arc = Arc::new(bytes);

        // Evict oldest if at capacity
        while inner.entries.len() >= inner.capacity {
            if let Some(old) = inner.order.pop_front() {
                inner.entries.remove(&old);
            } else {
                break;
            }
        }

        inner.entries.insert(path.to_path_buf(), Arc::clone(&arc));
        inner.order.push_back(path.to_path_buf());

        Ok(arc)
    }

    /// Number of distinct fonts currently cached.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Hit/miss stats for logging.
    pub fn stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.hits, inner.misses)
    }
}
