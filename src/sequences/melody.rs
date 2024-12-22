use crate::collections::traits::Collection;
use crate::default_methods;
use crate::entities::timing::DurationalEventTiming;
use crate::sequences::chord::ChordSeq;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct MelodyMember<T> {
    values: Vec<T>,
    timing: DurationalEventTiming,
    velocity: u8,
}

impl<T> Default for MelodyMember<T> {
    fn default() -> Self {
        MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            velocity: 64,
        }
    }
}

impl<T> From<T> for MelodyMember<T> {
    fn from(what: T) -> Self {
        MelodyMember {
            values: vec![what],
            ..Default::default()
        }
    }
}

impl<T> From<Vec<T>> for MelodyMember<T> {
    fn from(what: Vec<T>) -> Self {
        MelodyMember {
            values: what,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Melody<T> {
    contents: Vec<MelodyMember<T>>,
}

#[derive(Debug, PartialEq)]
pub enum MelodyError {
    InvalidValues,
}

impl<T> TryFrom<Vec<Vec<T>>> for Melody<T>
where
    T: Num + Debug + PartialOrd,
{
    type Error = MelodyError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.into_iter().map(|v| MelodyMember::from(v)).collect(),
        })
    }
}

impl<T> TryFrom<Vec<T>> for Melody<T>
where
    T: Num + Debug + PartialOrd,
{
    type Error = MelodyError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.into_iter().map(|v| MelodyMember::from(v)).collect(),
        })
    }
}

macro_rules! try_from_seq {
    ($type:ty) => {
        impl<T> TryFrom<$type> for Melody<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = MelodyError;

            fn try_from(what: $type) -> Result<Self, Self::Error> {
                Ok(Self {
                    contents: what
                        .to_pitches()
                        .into_iter()
                        .map(|v| MelodyMember::from(v))
                        .collect(),
                })
            }
        }
    };
}

try_from_seq!(NumericSeq<T>);
try_from_seq!(NoteSeq<T>);
try_from_seq!(ChordSeq<T>);

impl<T: Clone + Num + Debug + PartialOrd + Bounded> Collection<MelodyMember<T>> for Melody<T> {
    default_methods!(MelodyMember<T>);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>>
    Sequence<MelodyMember<T>, T> for Melody<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        for v in self.contents.iter_mut() {
            for p in v.values.iter_mut() {
                f(p);
            }
        }

        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents
            .iter()
            .flat_map(|v| v.values.iter())
            .copied()
            .collect()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents.iter().map(|v| v.values.clone()).collect()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>, String> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                1 => Ok(v.values[0]),
                _ => Err("must contain only one value".to_string()),
            })
            .collect()
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, String> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                0 => Ok(None),
                1 => Ok(Some(v.values[0])),
                _ => Err("must contain zero or one values".to_string()),
            })
            .collect()
    }
}
