use crate::algorithms;
use crate::collections::traits::Collection;
use crate::entities::scale::Scale;
use anyhow::{anyhow, Result};
use num_traits::{Num, PrimInt};
use std::fmt::Debug;
use std::iter::{zip, Sum};
use std::ops::SubAssign;
use std::slice::Iter;

#[macro_export]
macro_rules! default_sequence_methods {
    ($type:ty) => {
        fn new(contents: Vec<$type>) -> Self {
            Self {
                contents,
                metadata: MetadataList::default(),
            }
        }

        fn construct(&self, contents: Vec<$type>) -> Self {
            Self {
                contents,
                metadata: self.metadata.clone(),
            }
        }
    };
}

pub trait Sequence<
    T: Clone + Debug,
    PitchType: Clone + Copy + Debug + Num + PartialOrd + Sum + From<i32>,
>: Collection<T>
{
    fn mutate_pitches<F: Fn(&mut PitchType)>(self, f: F) -> Self;
    fn to_flat_pitches(&self) -> Vec<PitchType>;
    fn to_pitches(&self) -> Vec<Vec<PitchType>>;
    fn to_numeric_values(&self) -> Result<Vec<PitchType>>;
    fn to_optional_numeric_values(&self) -> Result<Vec<Option<PitchType>>>;

    fn min_value(&self) -> Option<PitchType> {
        self.to_flat_pitches()
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    fn max_value(&self) -> Option<PitchType> {
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
        if let (Some(min), Some(max)) = (self.min_value(), self.max_value()) {
            Some(max - min)
        } else {
            None
        }
    }

    fn find_if_window(&self, len: usize, step: usize, f: fn(&[T]) -> bool) -> Vec<usize> {
        let c = self.cts_ref();

        (0..=c.len() - len)
            .step_by(step)
            .filter(|i| f(&c[*i..*i + len]))
            .collect()
    }

    fn find_if_reverse_window(&self, len: usize, step: usize, f: fn(&[T]) -> bool) -> Vec<usize> {
        let c = self.cts_ref();
        let maxposs = self.length() - len;

        (0..=c.len() - len)
            .step_by(step)
            .map(|i| maxposs - i)
            .filter(|i| f(&c[*i..*i + len]))
            .collect()
    }

    fn transpose(self, t: PitchType) -> Result<Self> {
        Ok(self.mutate_pitches(algorithms::transpose(&t)))
    }

    fn transpose_to_min(self, t: PitchType) -> Result<Self> {
        match self.min_value() {
            Some(m) => self.transpose(t - m),
            None => Ok(self),
        }
    }

    fn transpose_to_max(self, t: PitchType) -> Result<Self> {
        match self.max_value() {
            Some(m) => self.transpose(t - m),
            None => Ok(self),
        }
    }

    fn invert(self, t: PitchType) -> Result<Self> {
        Ok(self.mutate_pitches(algorithms::invert(&t)))
    }

    fn augment<MT>(self, t: MT) -> Result<Self>
    where
        MT: algorithms::AugDim<PitchType>,
    {
        Ok(self.mutate_pitches(algorithms::augment(&t)))
    }

    fn diminish<MT>(self, t: MT) -> Result<Self>
    where
        MT: algorithms::AugDim<PitchType> + Num,
    {
        match algorithms::diminish(&t) {
            Ok(f) => Ok(self.mutate_pitches(f)),
            Err(e) => Err(anyhow!(e)),
        }
    }

    fn modulus(self, t: PitchType) -> Result<Self> {
        match algorithms::modulus(&t) {
            Ok(f) => Ok(self.mutate_pitches(f)),
            Err(e) => Err(anyhow!(e)),
        }
    }

    fn trim(self, min: PitchType, max: PitchType) -> Result<Self> {
        match algorithms::trim(Some(&min), Some(&max)) {
            Ok(f) => Ok(self.mutate_pitches(f)),
            Err(e) => Err(anyhow!(e)),
        }
    }

    fn trim_min(self, min: PitchType) -> Result<Self> {
        Ok(self.mutate_pitches(|v| {
            if *v < min {
                *v = min;
            }
        }))
    }

    fn trim_max(self, max: PitchType) -> Result<Self, String> {
        Ok(self.mutate_pitches(|v| {
            if *v > max {
                *v = max;
            }
        }))
    }

    fn bounce(self, min: PitchType, max: PitchType) -> Result<Self, String> {
        let diff = max - min;

        if diff < PitchType::from(0) {
            return Err(format!(
                "min {:?} must not be higher than max {:?}",
                min, max
            ));
        }

        Ok(self.mutate_pitches(|v| {
            if *v < min {
                let mut modulus = (min - *v) % (diff + diff);

                if modulus > diff {
                    modulus = diff + diff - modulus;
                }

                *v = min + modulus;
            } else if *v > max {
                let mut modulus = (*v - max) % (diff + diff);

                if modulus > diff {
                    modulus = diff + diff - modulus;
                }

                *v = max - modulus;
            }
        }))
    }

    fn bounce_min(self, min: PitchType) -> Result<Self, String> {
        Ok(self.mutate_pitches(|v| {
            if *v < min {
                *v = min + min - *v;
            }
        }))
    }

    fn bounce_max(self, max: PitchType) -> Result<Self, String> {
        Ok(self.mutate_pitches(|v| {
            if *v > max {
                *v = max + max - *v;
            }
        }))
    }

    fn filter_in_position(self, f: fn(&T) -> bool, default: T) -> Result<Self, String> {
        Ok(self.mutate_each(|m| {
            if !f(m) {
                *m = default.clone();
            }
        }))
    }

    fn collect_windows(&self, len: usize, step: usize) -> Vec<Vec<T>> {
        let c = self.cts_ref();
        let max = c.len() - len;

        (0..=max)
            .step_by(step)
            .map(move |i| c[i..i + len].to_vec())
            .collect()
    }

    fn flat_map_windows(
        self,
        len: usize,
        step: usize,
        f: fn(Vec<T>) -> Vec<T>,
    ) -> Result<Self, String> {
        let cts = self
            .collect_windows(len, step)
            .into_iter()
            .flat_map(f)
            .collect();

        Ok(self.with_contents(cts))
    }

    fn filter_windows(
        self,
        len: usize,
        step: usize,
        f: fn(Vec<T>) -> bool,
    ) -> Result<Self, String> {
        let cts: Vec<Vec<T>> = self
            .collect_windows(len, step)
            .into_iter()
            .filter(|v| f(v.clone()))
            .collect();

        let ret = cts.into_iter().flatten().collect();

        Ok(self.with_contents(ret))
    }

    fn pad(self, val: T, num: usize) -> Self {
        self.mutate_contents(|c| {
            c.splice(0..0, std::iter::repeat(val).take(num));
        })
    }

    fn scale(self, scale: Scale<PitchType>, zeroval: PitchType) -> Result<Self, String>
    where
        PitchType: PrimInt
            + From<i32>
            + From<i8>
            + TryFrom<usize>
            + TryInto<usize>
            + Debug
            + Num
            + Sum
            + SubAssign
            + num_traits::Euclid,
    {
        Ok(self.mutate_pitches(scale.fit_to_scale(&zeroval)))
    }

    fn combine(self, f: impl Fn((&T, &T)) -> T, seq: Self) -> Result<Self> {
        if self.length() != seq.length() {
            return Err(anyhow!(
                "sequence lengths ({} vs {}) were different",
                self.length(),
                seq.length()
            ));
        }

        let ret = zip(self.cts_ref(), seq.cts_ref()).map(f).collect();

        Ok(self.with_contents(ret))
    }

    fn map_with(self, f: impl Fn(Vec<&T>) -> T, seq: Vec<Self>) -> Result<Self> {
        let len = self.length();

        if seq.iter().any(|s| self.length() != s.length()) {
            return Err(anyhow!("sequence lengths were different"));
        }

        let mut iters: Vec<Iter<'_, T>> = seq.iter().map(|v| v.cts_ref().iter()).collect();
        iters.insert(0, self.cts_ref().iter());

        let ret: Vec<T> = (0..len)
            .map(|_| f(iters.iter_mut().map(|n| n.next().unwrap()).collect()))
            .collect();

        Ok(self.with_contents(ret))
    }

    fn map_with_indexed(self, f: impl Fn((usize, Vec<&T>)) -> T, seq: Vec<Self>) -> Result<Self> {
        let len = self.length();

        if seq.iter().any(|s| self.length() != s.length()) {
            return Err(anyhow!("sequence lengths were different"));
        }

        let mut iters: Vec<Iter<'_, T>> = seq.iter().map(|v| v.cts_ref().iter()).collect();
        iters.insert(0, self.cts_ref().iter());

        let ret: Vec<T> = (0..len)
            .map(|i| f((i, iters.iter_mut().map(|n| n.next().unwrap()).collect())))
            .collect();

        Ok(self.with_contents(ret))
    }
}
