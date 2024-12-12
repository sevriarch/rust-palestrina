// Traits common to all collections within Palestrina

use std::ops::Add;

#[derive(Clone)]
pub struct Collection<T: Clone + Copy> {
    contents: Vec<T>,
}

pub trait CollectionMethods<T: Clone + Copy> {
    fn construct(&self, contents: Vec<T>) -> Self;
    fn filter(&self, f: fn(T) -> bool) -> Self;
    fn map(&self, f: fn(T) -> T) -> Self;
}

impl<T: Clone + Copy> Collection<T> {
    pub fn new(contents: Vec<T>) -> Self {
        Self { contents }
    }
}

impl<T: Clone + Copy> CollectionMethods<T> for Collection<T> {
    fn construct(&self, contents: Vec<T>) -> Self {
        let mut new = self.clone();
        new.contents = contents;
        new
    }

    fn filter(&self, f: fn(T) -> bool) -> Self {
        let newcontents = self
            .contents
            .clone()
            .into_iter()
            .filter(|m| f(*m))
            .collect();

        self.construct(newcontents)
    }

    fn map(&self, f: fn(T) -> T) -> Self {
        let newcontents = self.contents.clone().into_iter().map(f).collect();

        self.construct(newcontents)
    }
}

impl<T: Add<Output = T> + Copy> Add for Collection<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut newcontents = self.contents.clone();
        newcontents.append(&mut rhs.contents.clone());

        self.construct(newcontents)
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::{Collection, CollectionMethods};

    #[test]
    fn filter() {
        let coll = Collection::new(vec![0, 2, 3, 4, 5]);
        let new = coll.filter(|v| v % 2 == 0);

        assert_eq!(new.contents, vec![0, 2, 4]);
    }

    #[test]
    fn map() {
        let coll = Collection::new(vec![0, 2, 3, 4, 5]);
        let new = coll.map(|v| v * v);

        assert_eq!(new.contents, vec![0, 4, 9, 16, 25]);
    }

    #[test]
    fn add() {
        let c1 = Collection::new(vec![1, 5, 6]);
        let c2 = Collection::new(vec![2, 7, 8, 9]);
        let c3 = Collection::new(vec![1, 5, 6, 2, 7, 8, 9]);

        assert_eq!((c1 + c2).contents, c3.contents);
    }
}
