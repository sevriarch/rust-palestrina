use crate::collections::traits::Collection;
use crate::metadata::MetadataList;
use crate::sequences::melody::Melody;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use anyhow::{anyhow, Result};
use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct ChordSeq<T> {
    pub contents: Vec<Vec<T>>,
    pub metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum ChordSeqError {
    InvalidValues,
}

/*
 * TODO: is a less clunky version of this possible?
 *
#[macro_export]
macro_rules! chordseq_member {
    ([$($x:expr),*]) => (
        Vec::from([ $($x),* ])
    );

    ($x:expr) => (
        Vec::from([ $x ])
    );
}

#[macro_export]
macro_rules! chordseq {
    ($($item:tt),*) => (
        ChordSeq::new(vec![
            $(chordseq_member!($item)),*
        ])
    );
}
*/

#[macro_export]
macro_rules! chordseq {
    ($([$($x:expr),*]),*) => (
        ChordSeq::new(vec![ $(vec![ $($x),* ]),* ])
    );

    ($($x:expr),*) => (
        ChordSeq::new(vec![ $(vec![ $x ]),* ])
    );
}

impl<T> TryFrom<Vec<T>> for ChordSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<T>) -> Result<Self> {
        Ok(Self::new(what.iter().map(|v| vec![*v]).collect()))
    }
}

impl<T> TryFrom<Vec<Option<T>>> for ChordSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Option<T>>) -> Result<Self> {
        Ok(Self::new(
            what.iter()
                .map(|v| v.map_or_else(Vec::new, |v| vec![v]))
                .collect(),
        ))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for ChordSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self> {
        Ok(Self::new(what))
    }
}

macro_rules! try_from_seq {
    (for $($t:ty)*) => ($(
        impl<T> TryFrom<$t> for ChordSeq<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = anyhow::Error;

            fn try_from(what: $t) -> Result<Self> {
                Ok(Self{
                    contents: what.to_pitches(),
                    metadata: what.metadata
                })
            }
        }
    )*)
}

try_from_seq!(for NumericSeq<T> NoteSeq<T> Melody<T>);

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

    fn to_numeric_values(&self) -> Result<Vec<T>> {
        self.contents
            .iter()
            .map(|v| match v.len() {
                1 => Ok(v[0]),
                _ => Err(anyhow!("must contain only one value")),
            })
            .collect()
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>> {
        self.contents
            .iter()
            .map(|v| match v.len() {
                0 => Ok(None),
                1 => Ok(Some(v[0])),
                _ => Err(anyhow!("must contain zero or one values")),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::metadata::{MetadataData, MetadataList};
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn from_macro() {
        assert_eq!(
            chordseq![[1], [], [2, 3], [4]],
            ChordSeq::new(vec![vec![1], vec![], vec![2, 3], vec![4]])
        );
        assert_eq!(
            chordseq![1, 2, 3],
            ChordSeq::new(vec![vec![1], vec![2], vec![3]])
        );
    }

    #[test]
    fn try_from_vec() {
        assert_eq!(
            ChordSeq::try_from(vec![5, 12, 16]).unwrap(),
            chordseq![5, 12, 16]
        );
    }

    #[test]
    fn try_from_vec_of_options() {
        assert_eq!(
            ChordSeq::try_from(vec![Some(5), None, Some(12), Some(16)]).unwrap(),
            chordseq![[5], [], [12], [16]]
        );
    }

    #[test]
    fn try_from_vec_of_vecs() {
        assert_eq!(
            ChordSeq::try_from(vec![vec![5], vec![], vec![12, 16]]).unwrap(),
            chordseq![[5], [], [12, 16]]
        );
    }

    #[test]
    fn try_from_melody() {
        assert_eq!(
            ChordSeq::try_from(Melody {
                contents: vec![
                    MelodyMember::from(vec![5]),
                    MelodyMember::from(vec![]),
                    MelodyMember::from(vec![12, 16]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            ChordSeq {
                contents: vec![vec![5], vec![], vec![12, 16]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert_eq!(
            ChordSeq::try_from(NoteSeq {
                contents: vec![Some(5), None, Some(12), Some(16)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            ChordSeq {
                contents: vec![vec![5], vec![], vec![12], vec![16]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            ChordSeq::try_from(NumericSeq {
                contents: vec![5, 12, 16],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            ChordSeq {
                contents: vec![vec![5], vec![12], vec![16]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            chordseq![[5], [], [12, 16]].to_flat_pitches(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            chordseq![[5], [], [12, 16]].to_pitches(),
            vec![vec![5], vec![], vec![12, 16]]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert!(ChordSeq::<i32>::new(vec![vec![]])
            .to_numeric_values()
            .is_err());
        assert!(chordseq![[12, 16]].to_numeric_values().is_err());
        assert_eq!(
            chordseq![5, 12, 16].to_numeric_values().unwrap(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert!(chordseq![[12, 16]].to_optional_numeric_values().is_err());
        assert_eq!(
            chordseq![[5], [], [12], [16]]
                .to_optional_numeric_values()
                .unwrap(),
            vec![Some(5), None, Some(12), Some(16)]
        );
    }
}
