/*
use crate::collections::traits::Collection;
use crate::default_methods;
use crate::sequences::traits::Sequence;

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::{Debug, Display};
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct ChordSeq<T> {
    contents: Vec<Vec<T>>,
}

#[derive(Debug)]
pub enum NumSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for ChordSeq<T>
where
    T: Copy + Num + Debug + PartialOrd,
{
    type Error = NumSeqError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.iter().map(|v| vec![*v]).collect(),
        })
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for ChordSeq<T>
where
    T: Copy + Num + Debug + PartialOrd,
{
    type Error = NumSeqError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Ok(Self { contents: what })
    }
}

impl<T: Clone + Copy + Num + Debug + Display + PartialOrd + Bounded> Collection<&Vec<T>>
    for ChordSeq<T>
where
    T: Copy,
{
    default_methods!(Vec<T>);
}

impl<T: Clone + Copy + Num + Debug + Display + PartialOrd + Bounded + Sum + From<i32>>
    Sequence<&Vec<T>, T> for ChordSeq<T>
where
    Vec<T>: Copy,
{
    // TODO: this needs to be a method that modifies if needed
    fn mutate_pitches<F: Fn(&T) -> T>(mut self, f: F) -> Self {
        for v in self.contents.iter_mut() {
            for p in v.iter_mut() {
                *p = f(p);
            }
        }

        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .collect()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents.clone()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>, String> {
        self.contents
            .iter()
            .map(|v| match v.len() {
                1 => Ok(v[0]),
                _ => Err("must contain only one value".to_string()),
            })
            .collect()
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, String> {
        self.contents
            .iter()
            .map(|v| match v.len() {
                0 => Ok(None),
                1 => Ok(Some(v[0])),
                _ => Err("must contain zero or one values".to_string()),
            })
            .collect()
    }
}

mod tests {
    use crate::collections::traits::Collection;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            ChordSeq::new(vec![vec![], vec![5], vec![], vec![12, 16], vec![44]]).to_flat_pitches(),
            vec![5, 12, 16, 44]
        );
    }
}
*/
