use crate::collections::traits::Collection;
use crate::metadata::MetadataList;
use crate::ops::pitch::{AugDim, Pitch, PitchError};
use crate::sequences::chord::ChordSeq;
use crate::sequences::melody::Melody;
use crate::sequences::note::NoteSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use anyhow::{anyhow, Context, Result};
use num_traits::{Bounded, FromPrimitive, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[macro_export]
macro_rules! numseq {
    ($($x:expr),*) => (
        NumericSeq::new(vec![ $($x),* ])
    );
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericSeq<T>
where
    T: Pitch<T> + Copy,
{
    pub contents: Vec<T>,
    pub metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum NumSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for NumericSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<T>) -> Result<Self> {
        Ok(NumericSeq::new(what))
    }
}

impl<T> TryFrom<Vec<Option<T>>> for NumericSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Option<T>>) -> Result<Self> {
        let mut ret = Vec::with_capacity(what.len());

        for v in what {
            ret.push(v.context("cannot include non-values in a NumericSeq")?);
        }

        Ok(NumericSeq::new(ret))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for NumericSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self> {
        let len = what.len();
        let cts: Vec<T> = what
            .into_iter()
            .filter_map(|v| if v.len() == 1 { Some(v[0]) } else { None })
            .collect();

        if cts.len() != len {
            Err(anyhow!("invalid values in input"))
        } else {
            Ok(NumericSeq::new(cts))
        }
    }
}

macro_rules! try_from_seq {
    (for $($type:ty)*) => ($(
        impl<T> TryFrom<$type> for NumericSeq<T>
        where
            T: Pitch<T> + Copy + Clone + Num + Debug + FromPrimitive + PartialOrd + Bounded + Sum,
        {
            type Error = anyhow::Error;

            fn try_from(what: $type) -> Result<Self> {
                Ok(NumericSeq{
                    contents: what.to_numeric_values()?,
                    metadata: what.metadata
                })
            }
        }
    )*)
}

try_from_seq!(for NoteSeq<T> ChordSeq<T> Melody<T>);

macro_rules! impl_fns_for_seq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { *p = p.$fn(n); });
            self
        }
    )*)
}

impl<T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum> Collection<T>
    for NumericSeq<T>
{
    default_collection_methods!(T);
    default_sequence_methods!(T);
}

impl<T: Pitch<T> + Clone + Copy + Num + Debug + FromPrimitive + PartialOrd + Bounded + Sum>
    Sequence<T, T> for NumericSeq<T>
{
    impl_fns_for_seq!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        self.contents.iter_mut().for_each(f);
        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents.clone()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents.clone().into_iter().map(|p| vec![p]).collect()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>> {
        Ok(self.contents.clone())
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>> {
        Ok(self.contents.clone().into_iter().map(|p| Some(p)).collect())
    }

    fn map_pitch_enumerated<MapT: Fn((usize, &T)) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().enumerate().for_each(|(i, p)| {
            *p = f((i, p));
        });
        self
    }

    fn filter_map_pitch_enumerated<MapT: Fn((usize, &T)) -> Option<T>>(
        mut self,
        f: MapT,
    ) -> Result<Self> {
        for (i, p) in self.contents.iter_mut().enumerate() {
            *p = f((i, p)).ok_or(anyhow!(PitchError::RequiredPitchAbsent(format!(
                "at index {}",
                i
            ))))?;
        }
        Ok(self)
    }
}

impl<T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum> NumericSeq<T> {
    pub fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        let repval = match p.len() {
            0 => {
                return Err(anyhow!(PitchError::RequiredPitchAbsent(
                    "set_pitches()".to_string()
                )))
            }
            1 => p[0],
            _ => {
                return Err(anyhow!(PitchError::MultiplePitchesNotAllowed(
                    "set_pitches()".to_string()
                )))
            }
        };

        self.contents.iter_mut().for_each(|m| {
            *m = repval;
        });
        Ok(self)
    }

    pub fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().for_each(|p| {
            *p = f(p);
        });
        self
    }

    pub fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        if self.contents.iter().all(f) {
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "filter_pitch()".to_string()
            )))
        }
    }

    pub fn filter_pitch_enumerated<FilterT: Fn((usize, &T)) -> bool>(
        self,
        f: FilterT,
    ) -> Result<Self> {
        if self.contents.iter().enumerate().all(f) {
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "filter_pitch_enumerated()".to_string()
            )))
        }
    }

    pub fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
        for p in self.contents.iter_mut() {
            *p = f(p).ok_or(anyhow!(PitchError::RequiredPitchAbsent(
                "Pitch.filter_map_pitch()".to_string()
            )))?;
        }
        Ok(self)
    }

    pub fn filter_map_pitch_enumerated<MapT: Fn((usize, &T)) -> Option<T>>(
        mut self,
        f: MapT,
    ) -> Result<Self> {
        for (i, p) in self.contents.iter_mut().enumerate() {
            *p = f((i, p)).ok_or(anyhow!(PitchError::RequiredPitchAbsent(format!(
                "at index {}",
                i
            ))))?;
        }
        Ok(self)
    }

    pub fn augment_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|p| {
            *p = p.augment_pitch(n);
        });
        self
    }

    pub fn diminish_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|p| {
            *p = p.diminish_pitch(n);
        });
        self
    }

    pub fn trim(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|p| {
            *p = p.trim(first, second);
        });
        self
    }

    pub fn bounce(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|p| {
            *p = p.bounce(first, second);
        });
        self
    }

    pub fn is_silent(&self) -> bool {
        self.contents.iter().all(|m| m.is_silent())
    }
}
// equality methods: equals isSubsetOf isSupersetOf isTransformationOf isTranspositionOf isInversionOf isRetrogradeOf
// isRetrogradeInversionOf hasPeriodicity[Of]

// window replacement methods: replaceIfWindow replaceIfReverseWindow

// setSlice loop repeat dupe dedupe shuffle pad padTo padRight padRightTo withPitch withPitches withPitchesAt
// mapPitches/filterPitches???

// sort chop partitionInPosition groupByInPosition untwine twine combine flatcombine combineMin combineMax combineOr combineAnd
// mapWith filterWith exchangeValuesIf

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
        assert_eq!(numseq![], NumericSeq::<i32>::new(vec![]));
        assert_eq!(numseq![1, 2, 3], NumericSeq::new(vec![1, 2, 3]));
    }

    #[test]
    fn try_from_vec() {
        assert_eq!(
            NumericSeq::try_from(vec![1, 2, 3]).unwrap(),
            numseq![1, 2, 3]
        );
    }

    #[test]
    fn try_from_vec_of_options() {
        assert_eq!(
            NumericSeq::try_from(vec![Some(1), Some(2), Some(3)]).unwrap(),
            numseq![1, 2, 3]
        );

        assert!(NumericSeq::try_from(vec![Some(1), None, Some(2), Some(3)]).is_err());
    }

    #[test]
    fn try_from_vec_of_vecs() {
        assert_eq!(
            NumericSeq::try_from(vec![vec![1], vec![2], vec![3]]).unwrap(),
            numseq![1, 2, 3]
        );

        assert!(NumericSeq::try_from(vec![Vec::<i32>::new()]).is_err());
        assert!(NumericSeq::try_from(vec![vec![1, 2, 3]]).is_err());
    }

    #[test]
    fn try_from_melody() {
        assert!(NumericSeq::try_from(Melody::<i32>::try_from(vec![vec![]]).unwrap()).is_err());
        assert!(NumericSeq::try_from(Melody::try_from(vec![vec![1, 2, 3]]).unwrap()).is_err());

        assert_eq!(
            NumericSeq::try_from(Melody {
                contents: vec![
                    MelodyMember::from(vec![1]),
                    MelodyMember::from(vec![2]),
                    MelodyMember::from(vec![3]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NumericSeq {
                contents: vec![1, 2, 3],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert!(NumericSeq::try_from(ChordSeq::<i32>::new(vec![vec![]])).is_err());
        assert!(NumericSeq::try_from(ChordSeq::new(vec![vec![1, 2, 3]])).is_err());

        assert_eq!(
            NumericSeq::try_from(ChordSeq {
                contents: vec![vec![1], vec![2], vec![3]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NumericSeq {
                contents: vec![1, 2, 3],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert!(NumericSeq::try_from(NoteSeq::<i32>::new(vec![None])).is_err());

        assert_eq!(
            NumericSeq::try_from(NoteSeq {
                contents: vec![Some(1), Some(2), Some(3)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NumericSeq {
                contents: vec![1, 2, 3],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].to_pitches(),
            vec![vec![4], vec![2], vec![5], vec![6], vec![3]]
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].to_flat_pitches(),
            vec![4, 2, 5, 6, 3]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].to_numeric_values().unwrap(),
            vec![4, 2, 5, 6, 3]
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].to_optional_numeric_values().unwrap(),
            vec![Some(4), Some(2), Some(5), Some(6), Some(3)]
        );
    }

    #[test]
    fn transpose_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].transpose_pitch(10),
            numseq![14, 12, 15, 16, 13]
        );
    }

    #[test]
    fn invert_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].invert_pitch(10),
            numseq![16, 18, 15, 14, 17]
        );
    }

    #[test]
    fn modulus() {
        assert_eq!(numseq![4, 2, 5, 6, 3].modulus(5), numseq![4, 2, 0, 1, 3]);
    }

    #[test]
    fn trim_min() {
        assert_eq!(numseq![4, 2, 5, 6, 3].trim_min(4), numseq![4, 4, 5, 6, 4]);
    }

    #[test]
    fn trim_max() {
        assert_eq!(numseq![4, 2, 5, 6, 3].trim_max(4), numseq![4, 2, 4, 4, 3]);
    }

    #[test]
    fn bounce_min() {
        assert_eq!(numseq![4, 2, 5, 6, 3].bounce_min(4), numseq![4, 6, 5, 6, 5]);
    }

    #[test]
    fn bounce_max() {
        assert_eq!(numseq![4, 2, 5, 6, 3].bounce_max(4), numseq![4, 2, 3, 2, 3]);
    }

    #[test]
    fn set_pitches() {
        assert!(NumericSeq::<i32>::new(vec![]).set_pitches(vec![]).is_err());
        assert!(NumericSeq::<i32>::new(vec![])
            .set_pitches(vec![55, 66])
            .is_err());
        assert_eq!(
            NumericSeq::<i32>::new(vec![])
                .set_pitches(vec![55])
                .unwrap(),
            NumericSeq::<i32>::new(vec![])
        );
        assert_eq!(
            numseq![4, 2, 5, 6, 3].set_pitches(vec![55]).unwrap(),
            numseq![55, 55, 55, 55, 55]
        );
    }

    #[test]
    fn map_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].map_pitch(|p| p * 2),
            numseq![8, 4, 10, 12, 6]
        );
    }

    #[test]
    fn map_pitch_enumerated() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].map_pitch_enumerated(|(i, p)| p + i as i32),
            numseq![4, 3, 7, 9, 7]
        );
    }

    #[test]
    fn filter_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].filter_pitch(|p| *p > 0).unwrap(),
            numseq![4, 2, 5, 6, 3]
        );

        assert!(numseq![4, 2, 5, 6, 3].filter_pitch(|p| *p > 2).is_err())
    }

    #[test]
    fn filter_pitch_enumerated() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3]
                .filter_pitch_enumerated(|(i, p)| (p + i as i32) < 10)
                .unwrap(),
            numseq![4, 2, 5, 6, 3]
        );

        assert!(numseq![4, 2, 5, 6, 3]
            .filter_pitch_enumerated(|(i, p)| *p > i as i32)
            .is_err())
    }

    #[test]
    fn filter_map_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3]
                .filter_map_pitch(|p| if *p > 1 { Some(p * 2) } else { None })
                .unwrap(),
            numseq![8, 4, 10, 12, 6]
        );

        assert!(numseq![4, 2, 5, 6, 3]
            .filter_map_pitch(|p| if *p > 2 { Some(p * 2) } else { None })
            .is_err());
    }

    #[test]
    fn filter_map_pitch_enumerated() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3]
                .filter_map_pitch_enumerated(|(i, p)| if *p > 1 {
                    Some(p + i as i32)
                } else {
                    None
                })
                .unwrap(),
            numseq![4, 3, 7, 9, 7]
        );

        assert!(numseq![4, 2, 5, 6, 3]
            .filter_map_pitch_enumerated(|(i, p)| if *p > 2 { Some(p + i as i32) } else { None })
            .is_err());
    }

    #[test]
    fn augment_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].augment_pitch(2),
            numseq![8, 4, 10, 12, 6]
        );

        assert_eq!(
            numseq![4, 2, 5, 6, 3].augment_pitch(2.5),
            numseq![10, 5, 12, 15, 7]
        );
    }

    #[test]
    fn diminish_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].diminish_pitch(0.5),
            numseq![8, 4, 10, 12, 6]
        );

        assert_eq!(
            numseq![4, 2, 5, 6, 3].diminish_pitch(2),
            numseq![2, 1, 2, 3, 1]
        );
    }

    #[test]
    fn trim() {
        assert_eq!(numseq![4, 2, 5, 6, 3].trim(3, 5), numseq![4, 3, 5, 5, 3]);
    }

    #[test]
    fn bounce() {
        assert_eq!(
            numseq![34, 12, 25, 46, 3].bounce(20, 30),
            numseq![26, 28, 25, 26, 23]
        );
    }

    #[test]
    fn is_silent() {
        assert!(NumericSeq::<i32>::new(vec![]).is_silent());
        assert!(!numseq![4, 2, 5, 6, 3].is_silent());
    }
}
