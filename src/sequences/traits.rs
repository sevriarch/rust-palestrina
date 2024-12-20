use crate::collections::traits::Collection;
use crate::entities::scale::Scale;
use num_traits::Num;
use std::fmt::Debug;
use std::iter::Sum;

pub trait Sequence<
    T: Clone + Copy + Debug,
    PitchType: Clone + Copy + Num + PartialOrd + Sum + From<i32>,
>: Collection<T>
{
    fn mutate_pitches<F: Fn(&T) -> T>(self, f: F) -> Self;
    fn to_flat_pitches(&self) -> Vec<PitchType>;
    fn to_pitches(&self) -> Result<Vec<Vec<PitchType>>, &str>;
    fn to_numeric_values(&self) -> Result<Vec<PitchType>, &str>;
    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, &str>;
    /*
    fn find_if_window(&self, len: usize, step: usize, f: fn(&[T]) -> bool) -> Vec<usize>;
    fn find_if_reverse_window(&self, len: usize, step: usize, f: fn(&[T]) -> bool) -> Vec<usize>;
    fn map_pitches(self, f: impl Fn(T) -> T) -> Self;
    fn transpose(self, t: T) -> Result<Self, String>;
    fn transpose_to_min(self, t: T) -> Result<Self, String>;
    fn transpose_to_max(self, t: T) -> Result<Self, String>;
    fn invert(self, t: T) -> Result<Self, String>;
    fn augment(self, t: T) -> Result<Self, String>;
    fn diminish(self, t: T) -> Result<Self, String>;
    fn modulus(self, t: T) -> Result<Self, String>;
    fn trim(self, min: T, max: T) -> Result<Self, String>;
    fn trim_min(self, min: T) -> Result<Self, String>;
    fn trim_max(self, max: T) -> Result<Self, String>;
    fn bounce(self, min: T, max: T) -> Result<Self, String>;
    fn bounce_min(self, min: PitchType) -> Result<Self, String>;
    fn bounce_max(self, max: PitchType) -> Result<Self, String>;
    fn filter_in_position(self, f: fn(T) -> bool, default: T) -> Result<Self, String>;
    fn flat_map_windows(
        self,
        len: usize,
        step: usize,
        f: fn(Vec<T>) -> Vec<T>,
    ) -> Result<Self, String>;
    fn filter_windows(self, len: usize, step: usize, f: fn(Vec<T>) -> bool)
        -> Result<Self, String>;
    fn pad(self, val: T, num: usize) -> Self;
    fn scale(self, scale: Scale<PitchType>, zeroval: PitchType) -> Result<Self, String>;
    */

    fn min(&self) -> Option<PitchType> {
        self.to_flat_pitches()
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    fn max(&self) -> Option<PitchType> {
        self.to_flat_pitches()
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    fn total(&self) -> Option<PitchType> {
        Some(self.to_flat_pitches().iter().copied().sum())
    }

    fn mean(&self) -> Option<PitchType> {
        let pitches = self.to_flat_pitches();
        let len = pitches.len() as i32;
        let mut iter = pitches.iter();
        let first = iter.next()?;

        Some(iter.fold(*first, |acc, x| acc + *x) / PitchType::from(len))
    }

    fn range(&self) -> Option<PitchType> {
        if let (Some(min), Some(max)) = (self.min(), self.max()) {
            Some(max - min)
        } else {
            None
        }
    }
}
