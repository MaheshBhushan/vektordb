//! Reusable visited-set for graph traversal.
//!
//! A fresh `HashSet` per query is a large fraction of search cost at high
//! QPS. Instead: one `u32` stamp per node, bumped generation counter per
//! pass. Clearing is O(1) except when the counter wraps (every 2^32 passes),
//! where we zero the array once.

pub struct VisitedSet {
    stamps: Vec<u32>,
    generation: u32,
}

impl VisitedSet {
    pub fn new() -> Self {
        Self { stamps: Vec::new(), generation: 0 }
    }

    /// Start a new traversal over a graph of `n` nodes.
    pub fn begin_pass(&mut self, n: usize) {
        if self.stamps.len() < n {
            self.stamps.resize(n, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    /// Mark `id` visited. Returns `true` if it was already visited this pass.
    /// Grows on demand: with concurrent inserts, links can name nodes
    /// published after `begin_pass` took its size snapshot.
    #[inline]
    pub fn mark(&mut self, id: u64) -> bool {
        if id as usize >= self.stamps.len() {
            self.stamps.resize(id as usize + 1, 0);
        }
        let slot = &mut self.stamps[id as usize];
        if *slot == self.generation {
            true
        } else {
            *slot = self.generation;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_and_resets() {
        let mut v = VisitedSet::new();
        v.begin_pass(10);
        assert!(!v.mark(3));
        assert!(v.mark(3));
        assert!(!v.mark(9));
        v.begin_pass(10);
        assert!(!v.mark(3), "new pass must forget previous marks");
    }

    #[test]
    fn grows_with_graph() {
        let mut v = VisitedSet::new();
        v.begin_pass(2);
        assert!(!v.mark(1));
        v.begin_pass(100);
        assert!(!v.mark(99));
    }
}
