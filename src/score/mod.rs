use crate::collections::traits::Collection;
use crate::default_collection_methods;
use crate::metadata::list::MetadataList;
use crate::sequences::melody::Melody;
use crate::sequences::traits::Sequence;
use num_traits::{Bounded, Num};
use std::convert::From;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct Score<T>
where
    T: Clone + Copy + Debug,
{
    contents: Vec<Melody<T>>,
    metadata: MetadataList,
}

impl<T> From<Vec<Melody<T>>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    fn from(m: Vec<Melody<T>>) -> Self {
        Self::new(m)
    }
}

impl<T> From<Melody<T>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    fn from(m: Melody<T>) -> Self {
        Self::new(vec![m])
    }
}

impl<T: Clone + Copy + Debug> Collection<Melody<T>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    default_collection_methods!(Melody<T>);

    fn new(m: Vec<Melody<T>>) -> Self {
        Self {
            contents: m,
            metadata: MetadataList::default(),
        }
    }

    fn construct(&self, contents: Vec<Melody<T>>) -> Self {
        Self {
            contents,
            metadata: self.metadata.clone(),
        }
    }
}

impl<T> Score<T>
where
    T: Clone + Copy + Debug + PartialOrd + Bounded + From<i32> + Num + Sum,
{
    pub fn pitch_range(&self) -> Option<(T, T)> {
        let min = self
            .contents
            .iter()
            .filter_map(|m| m.min_value())
            .min_by(|a, b| a.partial_cmp(b).unwrap())?;
        let max = self
            .contents
            .iter()
            .filter_map(|m| m.max_value())
            .max_by(|a, b| a.partial_cmp(b).unwrap())?;

        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_range() {
        assert!(Score::new(vec![Melody::<i32>::new(vec![])])
            .pitch_range()
            .is_none());

        assert_eq!(
            Score::new(vec![
                Melody::try_from(vec![1, 2, 3, 4, 5]).unwrap(),
                Melody::<i32>::new(vec![]),
                Melody::try_from(vec![6, 7, 8]).unwrap(),
            ])
            .pitch_range(),
            Some((1, 8))
        );
    }
}
