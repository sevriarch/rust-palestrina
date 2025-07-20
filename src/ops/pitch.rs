use crate::ops::arithmetic::AugDim;
use anyhow::{anyhow, Result};
use num_traits::{Bounded, Num};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use thiserror::Error;

#[derive(Clone, Error, Debug)]
pub enum PitchError {
    #[error("{0} operation led to the absence of a required pitch")]
    RequiredPitchAbsent(String),
    #[error("{0} operation failed because multiple pitches were generated")]
    MultiplePitchesNotAllowed(String),
}

/// A trait that implements various operations that can be executed on pitches
/// or containers for pitches.
pub trait Pitch<T>
where
    T: Copy + Bounded + Num,
{
    /// Set a single pitch
    fn set_pitch(self, p: T) -> Self;

    /// Set zero or more pitches.
    fn set_pitches(self, p: Vec<T>) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Mutate each individual pitch within the entity.
    fn mutate_pitch<MutateT: Fn(&mut T)>(self, f: MutateT) -> Self;

    /// A map operation on each individual pitch within the entity.
    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self;

    /// A filter operation on each individual pitches.
    ///
    /// Returns an Error if pitches are filtered out when they are required.
    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// A map operation on individual pitches that removes pitches where the map
    /// operation returns None, but retains those where the map operation returns
    /// Some(pitch).
    ///
    /// Returns an Error if pitches are filtered out when they are required.
    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Transpose pitches up or down.
    fn transpose_pitch(self, n: T) -> Self;

    /// Invert pitches around a specific value.
    fn invert_pitch(self, n: T) -> Self;

    /// Multiply pitches by a specific value.
    ///
    /// This value does not have to be the same type as the pitches being multiplied
    /// (automatic conversion will be performed) and it can be an integer or a float.
    ///
    /// Note that in the case where pitches are of an integer type and the multiplier
    /// is of a floating point type, results will always be rounded down.
    fn augment_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self;

    /// Divide pitches by a specific value.
    ///
    /// This value does not have to be the same type as the pitches being divided
    /// (automatic conversion will be performed) and it can be an integer or a float.
    ///
    /// Notes:
    ///
    /// If the divisor is zero and any pitches are divided, the program will panic.
    ///
    /// In the case where pitches are of an integer type and the divisor is of a
    /// floating point type, results will always be rounded down.
    fn diminish_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self;

    /// Euclidean remainder of the value (ie: modulus that cannot be lower than zero).
    ///
    /// Note that if the argument passed is zero and any pitches are mutated, the
    /// program will panic.
    fn modulus(self, n: T) -> Self;

    /// Replace any pitches below the passed argument with the passed argument.
    fn trim_min(self, n: T) -> Self;

    /// Replace any pitches above the passed argument with the passed argument.
    fn trim_max(self, n: T) -> Self;

    /// Replace any pitches below the lower argument with the lower argument.
    /// Replace any pitches above the higher argument with the higher argument.
    fn trim(self, first: T, second: T) -> Self;

    /// Invert any pitches below the passed argument.
    fn bounce_min(self, n: T) -> Self;

    /// Invert any pitches above the passed argument.
    fn bounce_max(self, n: T) -> Self;

    /// If a pitch is not in between the two values passed, invert it between the
    /// two repeardly until it is. If the two values passed are identical, this
    /// will set all pitches to that value.
    fn bounce(self, first: T, second: T) -> Self;

    /// Is this pitch or pitch container silent?
    #[allow(clippy::wrong_self_convention)]
    fn is_silent(self) -> bool;
}

impl<T> Pitch<T> for T
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    #[allow(unused_assignments)]
    fn set_pitch(mut self, p: T) -> Self {
        self = p;
        self
    }

    #[allow(unused_assignments)]
    fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        match p.len() {
            0 => Err(anyhow!(PitchError::RequiredPitchAbsent(
                "set_pitches()".to_string()
            ))),
            1 => {
                self = p[0];
                Ok(self)
            }
            _ => Err(anyhow!(PitchError::MultiplePitchesNotAllowed(
                "set_pitches()".to_string()
            ))),
        }
    }

    fn mutate_pitch<MutateT: Fn(&mut T)>(mut self, f: MutateT) -> Self {
        f(&mut self);
        self
    }

    fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
        self = f(&self);
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        if f(&self) {
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "Pitch.filter_pitch()".to_string()
            )))
        }
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
        if let Some(p) = f(&self) {
            self = p;
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "Pitch.filter_map_pitch()".to_string()
            )))
        }
    }

    fn transpose_pitch(mut self, n: T) -> Self {
        self += n;
        self
    }

    fn invert_pitch(mut self, n: T) -> Self {
        self = n + n - self;
        self
    }

    fn augment_pitch<MT: AugDim<T> + Copy>(mut self, n: MT) -> Self {
        n.augment_target(&mut self);
        self
    }

    fn diminish_pitch<MT: AugDim<T> + Copy>(mut self, n: MT) -> Self {
        n.diminish_target(&mut self);
        self
    }

    fn modulus(mut self, n: T) -> Self {
        // Can't use n.abs() as it's not implemented for unsigned values
        let n = if n < T::zero() { T::zero() - n } else { n };
        let ret = self % n;

        self = if ret < T::zero() { ret + n } else { ret };
        self
    }

    fn trim_min(mut self, n: T) -> Self {
        if self < n {
            self = n;
        }
        self
    }

    fn trim_max(mut self, n: T) -> Self {
        if self > n {
            self = n;
        }
        self
    }

    fn trim(mut self, first: T, second: T) -> Self {
        let (min, max) = if first > second {
            (second, first)
        } else {
            (first, second)
        };

        if self < min {
            self = min;
        } else if self > max {
            self = max;
        }
        self
    }

    fn is_silent(self) -> bool {
        false
    }

    fn bounce_min(mut self, n: T) -> Self {
        if self < n {
            self = n + n - self;
        }
        self
    }

    fn bounce_max(mut self, n: T) -> Self {
        if self > n {
            self = n + n - self;
        }
        self
    }

    fn bounce(mut self, first: T, second: T) -> Self {
        let (min, max) = if first > second {
            (second, first)
        } else {
            (first, second)
        };

        let diff = max - min;

        if diff.is_zero() {
            self = min;
        } else if self < min {
            let mut modulus = (min - self) % (diff + diff);

            if modulus > diff {
                modulus = diff + diff - modulus;
            }

            self = min + modulus;
        } else if self > max {
            let mut modulus = (self - max) % (diff + diff);

            if modulus > diff {
                modulus = diff + diff - modulus;
            }

            self = max - modulus;
        }
        self
    }
}

impl<T> Pitch<T> for &mut T
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    fn set_pitch(self, p: T) -> Self {
        *self = p;
        self
    }

    #[allow(unused_assignments)]
    fn set_pitches(self, p: Vec<T>) -> Result<Self> {
        match p.len() {
            0 => Err(anyhow!(PitchError::RequiredPitchAbsent(
                "set_pitches()".to_string()
            ))),
            1 => {
                *self = p[0];
                Ok(self)
            }
            _ => Err(anyhow!(PitchError::MultiplePitchesNotAllowed(
                "set_pitches()".to_string()
            ))),
        }
    }

    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self {
        *self = f(self);
        self
    }

    fn mutate_pitch<MutateT: Fn(&mut T)>(self, f: MutateT) -> Self {
        f(self);
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        if f(self) {
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "Pitch.filter_pitch()".to_string()
            )))
        }
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self> {
        if let Some(p) = f(self) {
            *self = p;
            Ok(self)
        } else {
            Err(anyhow!(PitchError::RequiredPitchAbsent(
                "Pitch.filter_map_pitch()".to_string()
            )))
        }
    }

    fn transpose_pitch(self, n: T) -> Self {
        *self += n;
        self
    }

    fn invert_pitch(self, n: T) -> Self {
        *self = n + n - *self;
        self
    }

    fn augment_pitch<MT: AugDim<T> + Copy>(self, n: MT) -> Self {
        n.augment_target(self);
        self
    }

    fn diminish_pitch<MT: AugDim<T> + Copy>(self, n: MT) -> Self {
        n.diminish_target(self);
        self
    }

    fn modulus(self, n: T) -> Self {
        // Can't use n.abs() as it's not implemented for unsigned ints
        let n = if n < T::zero() { T::zero() - n } else { n };
        let ret = *self % n;

        *self = if ret < T::zero() { ret + n } else { ret };
        self
    }

    fn trim_min(self, n: T) -> Self {
        if *self < n {
            *self = n;
        }
        self
    }

    fn trim_max(self, n: T) -> Self {
        if *self > n {
            *self = n;
        }
        self
    }

    fn trim(self, first: T, second: T) -> Self {
        let (min, max) = if first > second {
            (second, first)
        } else {
            (first, second)
        };

        if *self < min {
            *self = min;
        } else if *self > max {
            *self = max;
        }
        self
    }

    fn is_silent(self) -> bool {
        false
    }

    fn bounce_min(self, n: T) -> Self {
        if *self < n {
            *self = n + n - *self;
        }
        self
    }

    fn bounce_max(self, n: T) -> Self {
        if *self > n {
            *self = n + n - *self;
        }
        self
    }

    fn bounce(self, first: T, second: T) -> Self {
        let (min, max) = if first > second {
            (second, first)
        } else {
            (first, second)
        };

        let diff = max - min;

        if diff.is_zero() {
            *self = min;
        } else if *self < min {
            let mut modulus = (min - *self) % (diff + diff);

            if modulus > diff {
                modulus = diff + diff - modulus;
            }

            *self = min + modulus;
        } else if *self > max {
            let mut modulus = (*self - max) % (diff + diff);

            if modulus > diff {
                modulus = diff + diff - modulus;
            }

            *self = max - modulus;
        }
        self
    }
}

/*
macro_rules! impl_fns_for_option {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(self, n: $ty) -> Self {
            self.map(|v| v.$fn(n))
        }
    )*)
}

impl<T> Pitch<T> for Option<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    impl_fns_for_option!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    #[allow(unused_assignments)]
    fn set_pitch(mut self, p: T) -> Self {
        self = Some(p);
        self
    }

    #[allow(unused_assignments)]
    fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        match p.len() {
            0 => self = None,
            1 => self = Some(p[0]),
            _ => {
                return Err(anyhow!(PitchError::MultiplePitchesNotAllowed(
                    "set_pitches()".to_string()
                )))
            }
        }

        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self {
        self.map(|p| f(&p))
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        Ok(self.filter(f))
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self> {
        Ok(self.and_then(|p| f(&p)))
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.map(|p| p.augment_pitch(n))
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.map(|p| p.diminish_pitch(n))
    }

    fn trim(self, first: T, last: T) -> Self {
        self.map(|p| p.trim(first, last))
    }

    fn bounce(self, first: T, last: T) -> Self {
        self.map(|p| p.bounce(first, last))
    }

    fn is_silent(self) -> bool {
        self.is_none()
    }
}
*/

macro_rules! impl_fns_for_mut_option {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(self, n: $ty) -> Self {
            if let Some(v) = self {
                *v = *v.$fn(n);
            }
            self
        }
    )*)
}

impl<T> Pitch<T> for &mut Option<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    impl_fns_for_mut_option!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn set_pitch(self, p: T) -> Self {
        *self = Some(p);
        self
    }

    #[allow(unused_assignments)]
    fn set_pitches(self, p: Vec<T>) -> Result<Self> {
        match p.len() {
            0 => *self = None,
            1 => *self = Some(p[0]),
            _ => {
                return Err(anyhow!(PitchError::MultiplePitchesNotAllowed(
                    "set_pitches()".to_string()
                )))
            }
        }

        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self {
        if let Some(p) = self {
            p.map_pitch(f);
        }
        self
    }

    fn mutate_pitch<MutateT: Fn(&mut T)>(self, f: MutateT) -> Self {
        if let Some(p) = self {
            f(p);
        }
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        if let Some(p) = self {
            if !f(p) {
                *self = None;
            }
        }
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self> {
        if let Some(p) = self {
            *self = f(p);
        }
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        if let Some(p) = self {
            p.augment_pitch(n);
        }
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        if let Some(p) = self {
            p.diminish_pitch(n);
        }
        self
    }

    fn trim(self, first: T, last: T) -> Self {
        if let Some(p) = self {
            p.trim(first, last);
        }
        self
    }

    fn bounce(self, first: T, last: T) -> Self {
        if let Some(p) = self {
            p.bounce(first, last);
        }
        self
    }

    fn is_silent(self) -> bool {
        self.is_none()
    }
}

/*
/// Probably does not need to be implemented for Vec<T>
macro_rules! impl_fns_for_vec {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.iter_mut().for_each(|p| { *p = *p.$fn(n); });
            self
        }
    )*)
}

impl<T> Pitch<T> for Vec<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    impl_fns_for_vec!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    #[allow(unused_assignments)]
    fn set_pitch(mut self, p: T) -> Self {
        self = vec![p];
        self
    }

    #[allow(unused_assignments)]
    fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        self = p;
        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
        self.iter_mut().for_each(|p| *p = f(p));
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(mut self, f: FilterT) -> Result<Self> {
        self = self.into_iter().filter(f).collect();
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
        self = self.into_iter().filter_map(|p| f(&p)).collect();
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.iter_mut().for_each(|p| {
            *p = *p.augment_pitch(n);
        });
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.iter_mut().for_each(|p| {
            *p = *p.diminish_pitch(n);
        });
        self
    }

    fn trim(mut self, first: T, last: T) -> Self {
        self.iter_mut().for_each(|p| {
            *p = *p.trim(first, last);
        });
        self
    }

    fn bounce(mut self, first: T, last: T) -> Self {
        self.iter_mut().for_each(|p| {
            *p = *p.bounce(first, last);
        });
        self
    }

    fn is_silent(self) -> bool {
        self.is_empty()
    }
}
*/

macro_rules! impl_fns_for_mut_vec {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(self, n: $ty) -> Self {
            self.iter_mut().for_each(|p| { *p = *p.$fn(n); });
            self
        }
    )*)
}

impl<T> Pitch<T> for &mut Vec<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
{
    impl_fns_for_mut_vec!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn set_pitch(self, p: T) -> Self {
        *self = vec![p];
        self
    }

    fn set_pitches(self, p: Vec<T>) -> Result<Self> {
        *self = p;
        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self {
        self.iter_mut().for_each(|p| {
            p.map_pitch(&f);
        });
        self
    }

    fn mutate_pitch<MutateT: Fn(&mut T)>(self, f: MutateT) -> Self {
        self.iter_mut().for_each(f);
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        *self = self.iter().filter(|p| f(*p)).copied().collect();
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self> {
        *self = self.iter().filter_map(f).collect();
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.iter_mut().for_each(|p| {
            p.augment_pitch(n);
        });
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.iter_mut().for_each(|p| {
            p.diminish_pitch(n);
        });
        self
    }

    fn trim(self, first: T, last: T) -> Self {
        self.iter_mut().for_each(|p| {
            p.trim(first, last);
        });
        self
    }

    fn bounce(self, first: T, last: T) -> Self {
        self.iter_mut().for_each(|p| {
            p.bounce(first, last);
        });
        self
    }

    fn is_silent(self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_pitch() {
        assert_eq!(5.set_pitch(3), 3);
        assert_eq!(None.set_pitch(3), &Some(3));
        assert_eq!(Some(5).set_pitch(3), &Some(3));
        assert_eq!(vec![3, 4, 5].set_pitch(3), &vec![3]);

        let t = &mut 5;
        t.set_pitch(3);
        assert_eq!(*t, 3);

        let t = &mut None;
        t.set_pitch(3);
        assert_eq!(*t, Some(3));

        let t = &mut Some(5);
        t.set_pitch(3);
        assert_eq!(*t, Some(3));

        let t = &mut vec![3, 4, 5];
        t.set_pitch(3);
        assert_eq!(*t, vec![3]);
    }

    #[test]
    fn set_pitches() {
        assert!(5.set_pitches(vec![]).is_err());
        assert!(5.set_pitches(vec![2, 4]).is_err());
        assert!(None.set_pitches(vec![2, 4]).is_err());

        assert_eq!(5.set_pitches(vec![3]).unwrap(), 3);
        assert_eq!(None.set_pitches(vec![3]).unwrap(), &Some(3));
        assert_eq!(Some(5).set_pitches(vec![3]).unwrap(), &Some(3));
        assert_eq!(vec![3, 4, 5].set_pitches(vec![3]).unwrap(), &vec![3]);

        let t = &mut 5;

        assert!(t.set_pitches(vec![]).is_err());
        assert!(t.set_pitches(vec![2, 4]).is_err());
        assert!(t.set_pitches(vec![3]).is_ok());
        assert_eq!(*t, 3);

        let t = &mut None;
        assert!(t.set_pitches(vec![2, 4]).is_err());
        assert!(t.set_pitches(vec![3]).is_ok());
        assert_eq!(*t, Some(3));

        let t = &mut Some(5);
        assert!(t.set_pitches(vec![]).is_ok());
        assert_eq!(*t, None);

        let t = &mut vec![3, 4, 5];
        assert!(t.set_pitches(vec![3]).is_ok());
        assert_eq!(*t, vec![3]);
    }

    #[test]
    fn map_pitch() {
        macro_rules! map_pitch_test {
            ($init:expr, $fn:expr, $ret:expr) => {
                assert_eq!($init.map_pitch($fn), $ret);
            };
        }

        map_pitch_test!(3, |v| v + 1, 4);
        map_pitch_test!(None, |v| v + 1, &None);
        map_pitch_test!(Some(3), |v| v + 1, &Some(4));
        map_pitch_test!(vec![3, 4, 5], |v| v + 1, &vec![4, 5, 6]);
    }

    #[test]
    fn filter_pitch() {
        macro_rules! filter_pitch_test {
            ($init:expr, $fn:expr, $ret:expr) => {
                assert_eq!($init.filter_pitch($fn).unwrap(), $ret);
            };
        }

        assert!(5.filter_pitch(|v| v % 2 == 0).is_err());
        filter_pitch_test!(6, |v| v % 2 == 0, 6);
        filter_pitch_test!(None, |v| v % 2 == 0, &None);
        filter_pitch_test!(Some(5), |v| v % 2 == 0, &None);
        filter_pitch_test!(Some(6), |v| v % 2 == 0, &Some(6));
        filter_pitch_test!(vec![5, 6, 7], |v| v % 2 == 0, &vec![6]);
    }

    #[test]
    fn filter_map_pitch() {
        macro_rules! filter_map_pitch_test {
            ($init:expr, $fn:expr, $ret:expr) => {
                assert_eq!($init.filter_map_pitch($fn).unwrap(), $ret);
            };
        }

        assert!(6.filter_map_pitch(|_| None).is_err());
        filter_map_pitch_test!(6, |v| if v % 2 == 0 { Some(v / 2) } else { None }, 3);
        filter_map_pitch_test!(None, |v| if v % 2 == 0 { Some(v / 2) } else { None }, &None);
        filter_map_pitch_test!(
            Some(5),
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            &None
        );
        filter_map_pitch_test!(
            Some(6),
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            &Some(3)
        );
        filter_map_pitch_test!(
            vec![5, 6, 7],
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            &vec![3]
        );
    }

    #[test]
    fn transpose_pitch() {
        macro_rules! transpose_pitch_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.transpose_pitch($arg), $ret);
            };
        }

        transpose_pitch_test!(5, 6, 11);
        transpose_pitch_test!(5.5, 6.0, 11.5);
        transpose_pitch_test!(None::<i32>, 6, &None);
        transpose_pitch_test!(Some(5), 6, &Some(11));
        transpose_pitch_test!(vec![4, 5, 6], 6, &vec![10, 11, 12]);
    }

    #[test]
    fn invert_pitch() {
        macro_rules! invert_pitch_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.invert_pitch($arg), $ret);
            };
        }

        invert_pitch_test!(5, 6, 7);
        invert_pitch_test!(5.5, 6.0, 6.5);
        invert_pitch_test!(None, 6, &None);
        invert_pitch_test!(Some(5), 6, &Some(7));
        invert_pitch_test!(vec![4, 5, 6], 6, &vec![8, 7, 6]);
    }

    #[test]
    fn augment_pitch() {
        macro_rules! augment_pitch_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.augment_pitch($arg), $ret);
            };
        }

        augment_pitch_test!(4, 2, 8);
        augment_pitch_test!(4, 2.5, 10);
        augment_pitch_test!(2.5, 3, 7.5);
        augment_pitch_test!(2.5, 3.0, 7.5);
        augment_pitch_test!(None::<i32>, 2, &None);
        augment_pitch_test!(None::<i32>, 2.5, &None);
        augment_pitch_test!(None::<f32>, 2, &None);
        augment_pitch_test!(None::<f32>, 2.5, &None);
        augment_pitch_test!(Some(4), 2, &Some(8));
        augment_pitch_test!(Some(4), 2.5, &Some(10));
        augment_pitch_test!(Some(2.5), 3, &Some(7.5));
        augment_pitch_test!(Some(2.5), 3.0, &Some(7.5));
        augment_pitch_test!(vec![4, 5, 7], 2, &vec![8, 10, 14]);
        augment_pitch_test!(vec![4, 5, 7], 2.5, &vec![10, 12, 17]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], 3, &vec![12.0, 16.5, 19.5]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], 3.0, &vec![12.0, 16.5, 19.5]);
    }

    #[test]
    fn diminish_pitch() {
        macro_rules! diminish_pitch_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.diminish_pitch($arg), $ret);
            };
        }

        diminish_pitch_test!(8, 2, 4);
        diminish_pitch_test!(10, 2.5, 4);
        diminish_pitch_test!(7.5, 3, 2.5);
        diminish_pitch_test!(7.5, 3.0, 2.5);
        diminish_pitch_test!(None::<i32>, 2, &None);
        diminish_pitch_test!(None::<i32>, 2.5, &None);
        diminish_pitch_test!(None::<f32>, 2, &None);
        diminish_pitch_test!(None::<f32>, 2.5, &None);
        diminish_pitch_test!(Some(8), 2, &Some(4));
        diminish_pitch_test!(Some(10), 2.5, &Some(4));
        diminish_pitch_test!(Some(7.5), 3, &Some(2.5));
        diminish_pitch_test!(Some(7.5), 3.0, &Some(2.5));
        diminish_pitch_test!(vec![8, 10, 14], 2, &vec![4, 5, 7]);
        diminish_pitch_test!(vec![10, 12, 17], 2.5, &vec![4, 4, 6]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], 3, &vec![4.0, 5.5, 6.5]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], 3.0, &vec![4.0, 5.5, 6.5]);
    }

    #[test]
    fn modulus() {
        macro_rules! modulus_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.modulus($arg), $ret);
            };
        }

        modulus_test!(5, 3, 2);
        modulus_test!(5, -3, 2);
        modulus_test!(-5, 3, 1);
        modulus_test!(-5, -3, 1);
        modulus_test!(6, 3, 0);
        modulus_test!(-6, 3, 0);
        modulus_test!(None, 3, &None);
        modulus_test!(Some(5), 3, &Some(2));
        modulus_test!(Some(-5), 3, &Some(1));
        modulus_test!(vec![5.0, -5.0, 6.0], 1.5, &vec![0.5, 1.0, 0.0]);
    }

    #[test]
    fn trim_min() {
        macro_rules! trim_min_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.trim_min($arg), $ret);
            };
        }

        trim_min_test!(5, 3, 5);
        trim_min_test!(3, 5, 5);
        trim_min_test!(None, 5, &None);
        trim_min_test!(Some(5), 3, &Some(5));
        trim_min_test!(Some(3), 5, &Some(5));
        trim_min_test!(vec![5.0, 3.0], 4.0, &vec![5.0, 4.0]);
    }

    #[test]
    fn trim_max() {
        macro_rules! trim_max_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.trim_max($arg), $ret);
            };
        }

        trim_max_test!(5, 3, 3);
        trim_max_test!(3, 5, 3);
        trim_max_test!(None, 5, &None);
        trim_max_test!(Some(5), 3, &Some(3));
        trim_max_test!(Some(3), 5, &Some(3));
        trim_max_test!(vec![5.0, 3.0], 4.0, &vec![4.0, 3.0]);
    }

    #[test]
    fn trim() {
        macro_rules! trim_test {
            ($init:expr, $first:expr, $second:expr, $ret:expr) => {
                assert_eq!($init.trim($first, $second), $ret);
            };
        }

        trim_test!(2, 3, 5, 3);
        trim_test!(4, 3, 5, 4);
        trim_test!(6, 3, 5, 5);
        trim_test!(None, 3, 5, &None);
        trim_test!(Some(2), 3, 5, &Some(3));
        trim_test!(Some(4), 3, 5, &Some(4));
        trim_test!(Some(6), 3, 5, &Some(5));
        trim_test!(vec![2, 4, 6], 3, 5, &vec![3, 4, 5]);
    }

    #[test]
    fn bounce_min() {
        macro_rules! bounce_min_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.bounce_min($arg), $ret);
            };
        }

        bounce_min_test!(5, 3, 5);
        bounce_min_test!(3, 5, 7);
        bounce_min_test!(None, 5, &None);
        bounce_min_test!(Some(5), 3, &Some(5));
        bounce_min_test!(Some(3), 5, &Some(7));
        bounce_min_test!(vec![5.0, 3.0], 4.0, &vec![5.0, 5.0]);
    }

    #[test]
    fn bounce_max() {
        macro_rules! bounce_max_test {
            ($init:expr, $arg:expr, $ret:expr) => {
                assert_eq!($init.bounce_max($arg), $ret);
            };
        }

        bounce_max_test!(5, 3, 1);
        bounce_max_test!(3, 5, 3);
        bounce_max_test!(None, 5, &None);
        bounce_max_test!(Some(5), 3, &Some(1));
        bounce_max_test!(Some(3), 5, &Some(3));
        bounce_max_test!(vec![5.0, 3.0], 4.0, &vec![3.0, 3.0]);
    }

    #[test]
    fn bounce() {
        macro_rules! bounce_test {
            ($init:expr, $arg1:expr, $arg2:expr, $ret:expr) => {
                assert_eq!($init.bounce($arg1, $arg2), $ret);
            };
        }

        bounce_test!(2, 3, 6, 4);
        bounce_test!(4, 3, 6, 4);
        bounce_test!(8, 3, 6, 4);
        bounce_test!(-1, 3, 6, 5);
        bounce_test!(11, 3, 6, 5);
        bounce_test!(11, 6, 6, 6);
        bounce_test!(None, 6, 3, &None);
        bounce_test!(Some(2), 6, 3, &Some(4));
        bounce_test!(Some(4), 6, 3, &Some(4));
        bounce_test!(Some(8), 6, 3, &Some(4));
        bounce_test!(Some(8), 6, 6, &Some(6));
        bounce_test!(vec![2, 4, 8, -1, 11], 3, 6, &vec![4, 4, 4, 5, 5]);
        bounce_test!(vec![2, 4, 8, -1, 11], 6, 6, &vec![6, 6, 6, 6, 6]);
    }

    #[test]
    fn is_silent() {
        assert!(!0.is_silent());
        assert!(!Some(0).is_silent());
        assert!(None::<i32>.is_silent());
        assert!(Vec::<i32>::new().is_silent());
        assert!(!vec![0].is_silent());
    }
}
