use crate::collections::traits::Collection;
use crate::metadata::MetadataList;
use crate::ops::pitch::{AugDim, Pitch, PitchError};
use crate::sequences::chord::ChordSeq;
use crate::sequences::melody::Melody;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use anyhow::{anyhow, Result};
use num_traits::{Bounded, FromPrimitive, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

#[derive(Clone, Debug, PartialEq)]
pub struct NoteSeq<T> {
    pub contents: Vec<Option<T>>,
    pub metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum NoteSeqError {
    InvalidValues,
}

#[macro_export]
macro_rules! noteseq {
    ($($x:expr),*) => (
        NoteSeq::new(vec![ $(Option::from($x)),* ])
    );
}

impl<T> TryFrom<Vec<T>> for NoteSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<T>) -> Result<Self> {
        Ok(Self::new(what.iter().map(|v| Some(*v)).collect()))
    }
}

impl<T> TryFrom<Vec<Option<T>>> for NoteSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Option<T>>) -> Result<Self> {
        Ok(Self::new(what))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for NoteSeq<T>
where
    T: Pitch<T> + Copy + Clone + Num + Debug + PartialOrd + Bounded + Sum,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self> {
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
            Err(anyhow!("invalid values in vault"))
        } else {
            Ok(Self::new(cts))
        }
    }
}

macro_rules! try_from_seq {
    (for $($type:ty)*) => ($(
        impl<T> TryFrom<$type> for NoteSeq<T>
        where
            T: Pitch<T> + Copy + Clone + Num + Debug + FromPrimitive + PartialOrd + Bounded + Sum,
        {
            type Error = anyhow::Error;

            fn try_from(what: $type) -> Result<Self> {
                Ok(Self {
                    contents: what.to_optional_numeric_values()?,
                    metadata: what.metadata
                })
            }
        }
    )*)
}

try_from_seq!(for NumericSeq<T> ChordSeq<T> Melody<T>);

impl<T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum> Collection<Option<T>>
    for NoteSeq<T>
{
    default_collection_methods!(Option<T>);
    default_sequence_methods!(Option<T>);
}

impl<T: Pitch<T> + Clone + Copy + Num + Debug + FromPrimitive + PartialOrd + Bounded + Sum>
    Sequence<Option<T>, T> for NoteSeq<T>
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

    fn map_pitch_enumerated<MapT: Fn((usize, &T)) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().enumerate().for_each(|(i, m)| {
            if let Some(p) = m {
                *p = f((i, p));
            }
        });
        self
    }

    fn filter_pitch_enumerated<FilterT: Fn((usize, &T)) -> bool>(
        mut self,
        f: FilterT,
    ) -> Result<Self> {
        self.contents.iter_mut().enumerate().for_each(|(i, m)| {
            if let Some(p) = m {
                if !f((i, p)) {
                    *m = None;
                }
            }
        });
        Ok(self)
    }

    fn filter_map_pitch_enumerated<MapT: Fn((usize, &T)) -> Option<T>>(
        mut self,
        f: MapT,
    ) -> Result<Self> {
        self.contents
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = p.map(|v| f((i, &v))).flatten());
        Ok(self)
    }
}

macro_rules! impl_fns_for_seq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|m| {
                if let Some(p) = m {
                    *p = *p.$fn(n);
                }
            });
            self
        }
    )*)
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum> Pitch<T>
    for NoteSeq<T>
{
    impl_fns_for_seq!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        let repval = match p.len() {
            0 => None,
            1 => Some(p[0]),
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

    fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                *p = f(p);
            }
        });
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(mut self, f: FilterT) -> Result<Self> {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                if !f(p) {
                    *m = None;
                }
            }
        });
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
        self.contents
            .iter_mut()
            .for_each(|p| *p = p.map(|v| f(&v)).flatten());
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                *p = *p.augment_pitch(n);
            }
        });
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                *p = *p.diminish_pitch(n);
            }
        });
        self
    }

    fn trim(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                *p = *p.trim(first, second);
            }
        });
        self
    }

    fn bounce(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|m| {
            if let Some(p) = m {
                *p = *p.bounce(first, second);
            }
        });
        self
    }

    fn is_silent(self) -> bool {
        self.contents.iter().all(|m| m.is_none())
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::metadata::{MetadataData, MetadataList};
    use crate::ops::pitch::Pitch;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn from_macro() {
        assert_eq!(
            noteseq![1, 2, None, 3],
            NoteSeq::new(vec![Some(1), Some(2), None, Some(3)]),
        );
    }

    #[test]
    fn try_from_vec() {
        assert_eq!(NoteSeq::try_from(vec![1, 2, 3]).unwrap(), noteseq![1, 2, 3]);
    }

    #[test]
    fn try_from_vec_of_options() {
        assert_eq!(
            NoteSeq::try_from(vec![Some(1), None, Some(2), Some(3)]).unwrap(),
            noteseq![1, None, 2, 3]
        );
    }

    #[test]
    fn try_from_vec_of_vecs() {
        assert!(NoteSeq::try_from(vec![vec![1, 2, 3]]).is_err());

        assert_eq!(
            NoteSeq::try_from(vec![vec![1], vec![], vec![2], vec![3]]).unwrap(),
            noteseq![1, None, 2, 3]
        );
    }

    #[test]
    fn try_from_melody() {
        assert!(NoteSeq::try_from(Melody::try_from(vec![vec![1, 2, 3]]).unwrap()).is_err());

        assert_eq!(
            NoteSeq::try_from(Melody {
                contents: vec![
                    MelodyMember::from(vec![1]),
                    MelodyMember::from(vec![]),
                    MelodyMember::from(vec![2]),
                    MelodyMember::from(vec![3]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NoteSeq {
                contents: vec![Some(1), None, Some(2), Some(3)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert!(NoteSeq::try_from(ChordSeq::new(vec![vec![1, 2, 3]])).is_err());

        assert_eq!(
            NoteSeq::try_from(ChordSeq {
                contents: vec![vec![1], vec![], vec![2], vec![3]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NoteSeq {
                contents: vec![Some(1), None, Some(2), Some(3)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            NoteSeq::try_from(NumericSeq {
                contents: vec![1, 2, 3],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            NoteSeq {
                contents: vec![Some(1), Some(2), Some(3)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(noteseq![1, None, 2, 3].to_flat_pitches(), vec![1, 2, 3]);
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            noteseq![1, None, 2, 3].to_pitches(),
            vec![vec![1], vec![], vec![2], vec![3]]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert!(noteseq![1, None, 2, 3].to_numeric_values().is_err());

        assert_eq!(
            noteseq![1, 2, 3].to_numeric_values().unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert_eq!(
            noteseq![1, None, 2, 3]
                .to_optional_numeric_values()
                .unwrap(),
            vec![Some(1), None, Some(2), Some(3)]
        );
    }

    #[test]
    fn set_pitches() {
        assert!(NoteSeq::<i32>::new(vec![])
            .set_pitches(vec![55, 66])
            .is_err());
        assert_eq!(
            NoteSeq::<i32>::new(vec![]).set_pitches(vec![55]).unwrap(),
            NoteSeq::<i32>::new(vec![])
        );
        assert_eq!(
            noteseq![4, 2, 5, 6, 3].set_pitches(vec![]).unwrap(),
            noteseq![None, None, None, None, None]
        );
        assert_eq!(
            noteseq![4, 2, 5, 6, 3].set_pitches(vec![55]).unwrap(),
            noteseq![55, 55, 55, 55, 55]
        );
    }
}
