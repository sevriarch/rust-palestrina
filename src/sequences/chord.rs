use crate::collections::traits::Collection;
use crate::metadata::list::MetadataList;
use crate::sequences::melody::Melody;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct ChordSeq<T> {
    contents: Vec<Vec<T>>,
    metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum ChordSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for ChordSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = ChordSeqError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self::new(what.iter().map(|v| vec![*v]).collect()))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for ChordSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = ChordSeqError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Ok(Self::new(what))
    }
}

macro_rules! try_from_seq {
    ($type:ty) => {
        impl<T> TryFrom<$type> for ChordSeq<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = ChordSeqError;

            fn try_from(what: $type) -> Result<Self, Self::Error> {
                Ok(Self::new(what.to_pitches()))
            }
        }
    };
}

try_from_seq!(Melody<T>);
try_from_seq!(NoteSeq<T>);
try_from_seq!(NumericSeq<T>);

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded> Collection<Vec<T>> for ChordSeq<T> {
    default_collection_methods!(Vec<T>);
    default_sequence_methods!(Vec<T>);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>> Sequence<Vec<T>, T>
    for ChordSeq<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        self.contents.iter_mut().for_each(|m| {
            for p in m.iter_mut() {
                f(p)
            }
        });
        self
    }

    fn mutate_pitches_ref<F: Fn(&mut T)>(&mut self, f: F) -> &Self {
        self.mutate_each(|m| {
            for p in m.iter_mut() {
                f(p)
            }
        })
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
            ChordSeq::try_from(vec![5, 12, 16]),
            Ok(ChordSeq::new(vec![vec![5], vec![12], vec![16]]))
        );
    }

    #[test]
    fn try_from_vec_vec() {
        assert_eq!(
            ChordSeq::try_from(vec![vec![5], vec![], vec![12, 16]]),
            Ok(ChordSeq::new(vec![vec![5], vec![], vec![12, 16]]))
        );
    }

    #[test]
    fn try_from_melody() {
        assert_eq!(
            ChordSeq::try_from(Melody::try_from(vec![vec![5], vec![], vec![12, 16]]).unwrap()),
            Ok(ChordSeq::new(vec![vec![5], vec![], vec![12, 16]]))
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            ChordSeq::try_from(NumericSeq::new(vec![5, 12, 16])),
            Ok(ChordSeq::new(vec![vec![5], vec![12], vec![16]]))
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert_eq!(
            ChordSeq::try_from(NoteSeq::new(vec![Some(5), None, Some(12), Some(16)])),
            Ok(ChordSeq::new(vec![vec![5], vec![], vec![12], vec![16]]))
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            ChordSeq::new(vec![vec![5], vec![], vec![12, 16]]).to_flat_pitches(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            ChordSeq::new(vec![vec![5], vec![], vec![12, 16]]).to_pitches(),
            vec![vec![5], vec![], vec![12, 16]]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert!(ChordSeq::<i32>::new(vec![vec![]])
            .to_numeric_values()
            .is_err());
        assert!(ChordSeq::new(vec![vec![12, 16]])
            .to_numeric_values()
            .is_err());
        assert_eq!(
            ChordSeq::new(vec![vec![5], vec![12], vec![16]]).to_numeric_values(),
            Ok(vec![5, 12, 16])
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert!(ChordSeq::new(vec![vec![12, 16]])
            .to_optional_numeric_values()
            .is_err());
        assert_eq!(
            ChordSeq::new(vec![vec![5], vec![], vec![12], vec![16]]).to_optional_numeric_values(),
            Ok(vec![Some(5), None, Some(12), Some(16)])
        );
    }
}
