use crate::collections::traits::Collection;
use crate::metadata::list::MetadataList;
use crate::sequences::chord::ChordSeq;
use crate::sequences::melody::Melody;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use anyhow::{anyhow, Result};
use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct NoteSeq<T> {
    pub contents: Vec<Option<T>>,
    pub metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum NoteSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for NoteSeq<T>
where
    T: Copy + Clone + Num + Debug + PartialOrd + Bounded,
{
    type Error = NoteSeqError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self::new(what.iter().map(|v| Some(*v)).collect()))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for NoteSeq<T>
where
    T: Copy + Clone + Num + Debug + PartialOrd + Bounded,
{
    type Error = NoteSeqError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        let len = what.len();
        let cts: Vec<Option<T>> = what
            .into_iter()
            .filter_map(|v| match v.len() {
                0 => Some(None),
                1 => Some(Some(v[0])),
                _ => None,
            })
            .collect();

        if cts.len() != len {
            Err(NoteSeqError::InvalidValues)
        } else {
            Ok(Self::new(cts))
        }
    }
}

macro_rules! try_from_seq {
    ($type:ty) => {
        impl<T> TryFrom<$type> for NoteSeq<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = NoteSeqError;

            fn try_from(what: $type) -> Result<Self, Self::Error> {
                match what.to_optional_numeric_values() {
                    Ok(vals) => Ok(Self::new(vals)),
                    Err(_) => Err(NoteSeqError::InvalidValues),
                }
            }
        }
    };
}

try_from_seq!(Melody<T>);
try_from_seq!(ChordSeq<T>);
try_from_seq!(NumericSeq<T>);

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded> Collection<Option<T>> for NoteSeq<T> {
    default_collection_methods!(Option<T>);
    default_sequence_methods!(Option<T>);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>> Sequence<Option<T>, T>
    for NoteSeq<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                f(p);
            }
        });
        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents.clone().into_iter().flatten().collect()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents
            .clone()
            .into_iter()
            .map(|p| match p {
                Some(v) => vec![v],
                None => vec![],
            })
            .collect()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>> {
        let vals = self.to_flat_pitches();

        if vals.len() != self.contents.len() {
            Err(anyhow!("at least one value was None"))
        } else {
            Ok(vals)
        }
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>> {
        Ok(self.contents.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::Melody;
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn try_from_vec() {
        assert_eq!(
            NoteSeq::try_from(vec![1, 2, 3]),
            Ok(NoteSeq::new(vec![Some(1), Some(2), Some(3)]))
        );
    }

    #[test]
    fn try_from_vec_vec() {
        assert!(NoteSeq::try_from(vec![vec![1, 2, 3]]).is_err());

        assert_eq!(
            NoteSeq::try_from(vec![vec![1], vec![], vec![2], vec![3]]),
            Ok(NoteSeq::new(vec![Some(1), None, Some(2), Some(3)]))
        );
    }

    #[test]
    fn try_from_melody() {
        assert!(NoteSeq::try_from(Melody::try_from(vec![vec![1, 2, 3]]).unwrap()).is_err());

        assert_eq!(
            NoteSeq::try_from(Melody::try_from(vec![vec![1], vec![], vec![2], vec![3]]).unwrap()),
            Ok(NoteSeq::new(vec![Some(1), None, Some(2), Some(3)]))
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert!(NoteSeq::try_from(ChordSeq::new(vec![vec![1, 2, 3]])).is_err());

        assert_eq!(
            NoteSeq::try_from(ChordSeq::new(vec![vec![1], vec![], vec![2], vec![3]])),
            Ok(NoteSeq::new(vec![Some(1), None, Some(2), Some(3)])),
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            NoteSeq::try_from(NumericSeq::new(vec![1, 2, 3])),
            Ok(NoteSeq::new(vec![Some(1), Some(2), Some(3)])),
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            NoteSeq::new(vec![Some(1), None, Some(2), Some(3)]).to_flat_pitches(),
            vec![1, 2, 3]
        );
    }
    #[test]
    fn to_pitches() {
        assert_eq!(
            NoteSeq::new(vec![Some(1), None, Some(2), Some(3)]).to_pitches(),
            vec![vec![1], vec![], vec![2], vec![3]]
        );
    }
    #[test]
    fn to_numeric_values() {
        assert!(NoteSeq::<i32>::new(vec![None]).to_numeric_values().is_err());

        assert_eq!(
            NoteSeq::new(vec![Some(1), Some(2), Some(3)])
                .to_numeric_values()
                .unwrap(),
            vec![1, 2, 3]
        );
    }
    #[test]
    fn to_optional_numeric_values() {
        assert_eq!(
            NoteSeq::new(vec![Some(1), None, Some(2), Some(3)])
                .to_optional_numeric_values()
                .unwrap(),
            vec![Some(1), None, Some(2), Some(3)]
        );
    }
}
