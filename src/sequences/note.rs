use crate::collections::traits::Collection;
use crate::default_methods;
use crate::sequences::traits::Sequence;

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::{Debug, Display};
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct NoteSeq<T> {
    contents: Vec<Option<T>>,
}

#[derive(Debug)]
pub enum NumSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for NoteSeq<T>
where
    T: Copy + Num + Debug + PartialOrd,
{
    type Error = NumSeqError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.iter().map(|v| Some(*v)).collect(),
        })
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for NoteSeq<T>
where
    T: Copy + Num + Debug + PartialOrd,
{
    type Error = NumSeqError;

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
            Err(NumSeqError::InvalidValues)
        } else {
            Ok(Self { contents: cts })
        }
    }
}

impl<T: Clone + Copy + Num + Debug + Display + PartialOrd + Bounded> Collection<Option<T>>
    for NoteSeq<T>
{
    default_methods!(Option<T>);
}

impl<T: Clone + Copy + Num + Debug + Display + PartialOrd + Bounded + Sum + From<i32>>
    Sequence<Option<T>, T> for NoteSeq<T>
where
    Vec<Option<T>>: Copy,
{
    // TODO: this needs to be a method that modifies if needed
    fn mutate_pitches<F: Fn(&T) -> T>(mut self, f: F) -> Self {
        for v in self.contents.iter_mut() {
            if let Some(val) = v {
                *v = Some(f(val));
            }
        }

        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents.into_iter().filter_map(|v| v).collect()
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

    fn to_numeric_values(&self) -> Result<Vec<T>, String> {
        let vals = self.to_flat_pitches();

        if vals.len() != self.contents.len() {
            Err("at least one value was None".to_string())
        } else {
            Ok(vals)
        }
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, String> {
        Ok(self.contents.clone())
    }
}
