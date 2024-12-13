// Traits common to all Sequences
use crate::collections::traits::Collection;

#[derive(Clone)]
pub struct Sequence<T: Clone + Copy> {
    contents: Vec<T>,
}

impl<T: Clone + Copy> Collection<T> for Sequence<T> {
    fn new(contents: Vec<T>) -> Self {
        Self { contents }
    }

    fn cts(&self) -> Vec<T> {
        self.contents.clone()
    }

    fn length(&self) -> usize {
        self.contents.len()
    }

    fn construct(&self, contents: Vec<T>) -> Box<Self> {
        let mut newseq = self.clone();
        newseq.contents = contents;
        Box::new(newseq)
    }

    fn clone_contents(&self) -> Vec<T> {
        self.contents.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::sequence::sequence::Sequence;

    #[test]
    fn filter() {
        let coll = Sequence::new(vec![0, 2, 3, 4, 5]);
        let new = coll.filter(|v| v % 2 == 0);

        assert_eq!(new.contents, vec![0, 2, 4]);
    }

    #[test]
    fn map() {
        let coll = Sequence::new(vec![0, 2, 3, 4, 5]);
        let new = coll.map(|v| v * v);

        assert_eq!(new.contents, vec![0, 4, 9, 16, 25]);
    }

    /*
    #[test]
    fn add() {
        let c1 = Sequence::new(vec![1, 5, 6]);
        let c2 = Sequence::new(vec![2, 7, 8, 9]);

        let c3 = Sequence::new(vec![1, 5, 6, 2, 7, 8, 9]);

        assert_eq!((c1 + c2).contents, c3.contents);
    }
    */
}
