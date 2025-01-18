use anyhow::{anyhow, Result};
use num_traits::{Euclid, Num};
use std::cmp::PartialOrd;
use std::ops::SubAssign;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum ScaleError {
    #[error("Unknown scale: {0}")]
    UnknownScale(String),
    #[error("Scale cannot be empty")]
    EmptyScale,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scale<T>
where
    T: Copy + Num + From<i8> + TryInto<usize> + TryFrom<usize> + Euclid + PartialOrd + SubAssign,
{
    notes: Vec<T>,
    octave: T,
}

impl<T> Scale<T>
where
    T: Copy + Num + From<i8> + TryInto<usize> + TryFrom<usize> + Euclid + PartialOrd + SubAssign,
{
    pub fn new(notes: Vec<T>, octave: T) -> Self {
        Self { notes, octave }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        Ok(Self {
            notes: Scale::name_to_notes(name)?,
            octave: T::from(12),
        })
    }

    pub fn with_octave(mut self, o: T) -> Self {
        self.octave = o;
        self
    }

    pub fn with_notes(mut self, notes: Vec<T>) -> Result<Self> {
        self.notes = notes;
        Ok(self)
    }

    pub fn with_name(self, name: &str) -> Result<Self> {
        self.with_notes(Scale::name_to_notes(name)?)
    }

    // TODO: planned refactor will eliminate the need for this
    #[allow(clippy::type_complexity)]
    pub fn fit_to_scale<'a>(&'a self, zeroval: &'a T) -> Result<Box<dyn Fn(&mut T) + 'a>> {
        let len: T = match self.notes.len().try_into() {
            Ok(v) => Ok(v),
            Err(_) => Err(anyhow!(ScaleError::EmptyScale)),
        }?;

        Ok(Box::new(move |v| {
            let ix = v.rem_euclid(&len);
            let mut octaves = *v / len;

            if !ix.is_zero() && *v < T::zero() {
                octaves -= T::one();
            }

            // This should never happen, so defaulting to 0 should be safe
            let ix = ix.try_into().unwrap_or(0);

            *v = *zeroval + self.notes[ix] + octaves * self.octave;
        }))
    }

    pub fn name_to_notes(name: &str) -> Result<Vec<T>> {
        let notes: Vec<i8> = match name {
            "chromatic" => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            "octatonic12" => Ok(vec![0, 1, 3, 4, 6, 7, 9, 10]),
            "octatonic21" => Ok(vec![0, 2, 3, 5, 6, 8, 9, 11]),
            "wholetone" => Ok(vec![0, 2, 4, 6, 8, 10]),
            "major" => Ok(vec![0, 2, 4, 5, 7, 9, 11]),
            "minor" => Ok(vec![0, 2, 3, 5, 7, 8, 10]),
            "ionian" => Ok(vec![0, 2, 4, 5, 7, 9, 11]),
            "dorian" => Ok(vec![0, 2, 3, 5, 7, 9, 10]),
            "phrygian" => Ok(vec![0, 1, 3, 5, 7, 8, 10]),
            "lydian" => Ok(vec![0, 2, 4, 6, 7, 9, 11]),
            "mixolydian" => Ok(vec![0, 2, 4, 5, 7, 9, 10]),
            "aeolian" => Ok(vec![0, 2, 3, 5, 7, 8, 10]),
            "locrian" => Ok(vec![0, 1, 3, 5, 6, 8, 10]),
            "pentatonic" => Ok(vec![0, 2, 4, 7, 9]),
            "pentatonicc" => Ok(vec![0, 2, 4, 7, 9]),
            "pentatonicd" => Ok(vec![0, 2, 5, 7, 10]),
            "pentatonice" => Ok(vec![0, 3, 5, 8, 10]),
            "pentatonicg" => Ok(vec![0, 2, 5, 7, 9]),
            "pentatonica" => Ok(vec![0, 3, 5, 7, 10]),
            _ => Err(anyhow!(ScaleError::UnknownScale(name.to_string()))),
        }?;

        Ok(notes.into_iter().map(T::from).collect())
    }
}

#[macro_export]
macro_rules! scale {
    ([$num:expr]) => {{
        Ok(Scale::new($num.to_vec(), 12))
    }};

    ([$num:expr], $octave:expr) => {{
        Ok(Scale::new($num.to_vec(), $octave))
    }};

    ($name:expr) => {
        Scale::name_to_notes($name).map(|n| Scale::new(n, 12))
    };

    ($name:expr, $octave:expr) => {
        Scale::name_to_notes($name).map(|n| Scale::new(n, $octave))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_macro() {
        assert!(scale!["turbofish"].is_err());
        assert!(scale!["turbofish", 10].is_err());

        assert_eq!(
            scale!["lydian"].unwrap(),
            Scale {
                notes: vec![0, 2, 4, 6, 7, 9, 11],
                octave: 12
            }
        );

        assert_eq!(
            scale!["pentatonic", 10].unwrap(),
            Scale {
                notes: vec![0, 2, 4, 7, 9],
                octave: 10
            }
        );

        let s: Result<Scale<i32>, anyhow::Error> = scale![[[0, 2, 4, 6, 8, 10]]];
        assert_eq!(
            s.unwrap(),
            Scale {
                notes: vec![0, 2, 4, 6, 8, 10],
                octave: 12
            }
        );

        let s: Result<Scale<i32>, anyhow::Error> = scale![[[0, 2, 4, 6, 8, 10]], 24];
        assert_eq!(
            s.unwrap(),
            Scale {
                notes: vec![0, 2, 4, 6, 8, 10],
                octave: 24
            }
        );
    }

    #[test]
    fn fit_to_scale() {
        let scale = scale!["lydian"].unwrap();

        let mut vec: Vec<i32> = (-20_i32..20_i32).collect::<Vec<i32>>();
        let f = scale.fit_to_scale(&60).unwrap();

        for v in vec.iter_mut() {
            f(v);
        }

        assert_eq!(
            vec,
            vec![
                26, 28, 30, 31, 33, 35, 36, 38, 40, 42, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60,
                62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91, 93
            ]
        );

        let mut vec: Vec<i32> = vec![0, 2, 4];
        let f = scale.fit_to_scale(&0).unwrap();

        for v in vec.iter_mut() {
            f(v);
        }

        assert_eq!(vec, vec![0, 4, 7]);
    }
}
