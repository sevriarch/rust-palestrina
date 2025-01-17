use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
};
use anyhow::{anyhow, Result};
use num_traits::Zero;
use thiserror::Error;

pub trait AugDim<MT>
where
    MT: Copy,
{
    fn augment_target(&self, v: &mut MT);
    fn diminish_target(&self, v: &mut MT);
}

macro_rules! single_conv_aug_dim {
    ($type:ident for $($target:ty)*) => ($(
        impl AugDim<$target> for $type {
            fn augment_target(&self, v: &mut $target) {
                *v *= (*self as $target);
            }

            fn diminish_target(&self, v: &mut $target) {
                *v /= (*self as $target);
            }
        }
    )*)
}

macro_rules! double_conv_aug_dim {
    ($type:ident for $($target:ty)*) => ($(
        impl AugDim<$target> for $type {
            fn augment_target(&self, v: &mut $target) {
                *v = ((*v as $type) * self) as $target;
            }

            fn diminish_target(&self, v: &mut $target) {
                *v = ((*v as $type) / self) as $target;
            }
        }
    )*)
}

macro_rules! make_double_conv_aug_dim {
    (for $($ty:ident)*) => ($(
        double_conv_aug_dim!($ty for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
    )*)
}

macro_rules! make_single_conv_aug_dim_int {
    (for $($ty:ident)*) => ($(
        single_conv_aug_dim!($ty for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
    )*)
}

macro_rules! make_single_conv_aug_dim_float {
    (for $($ty:ident)*) => ($(
        single_conv_aug_dim!($ty for f32 f64);
    )*)
}

make_single_conv_aug_dim_int!(for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
make_single_conv_aug_dim_float!(for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64);
make_double_conv_aug_dim!(for f32 f64);

#[derive(Clone, Error, Debug)]
pub enum PitchError {
    #[error("{0} operation generated no pitches when a pitch was required")]
    RequiredPitchAbsent(String),
}

/// A trait that implements various operations that can be executed on pitches
/// or containers for pitches.
pub trait Pitch<T>
where
    T: Copy,
{
    /// A map operation on pitches.
    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self;

    /// A map operation on pitches where the pitch is passed to the mapper function
    /// in a tuple of (position, pitch). The position passed is an Option<usize>, which
    /// is always None when the map operation is taking place outside of a Sequence
    /// context, and will contain the position in the Sequence if the map operation
    /// takes place in a Sequence context.
    ///
    /// Hence, there is no reason to use this mathod outside of a Sequence context.
    fn map_pitch_enumerated<MapT: Fn((Option<usize>, &T)) -> T>(self, f: MapT) -> Self;

    /// A map operation on pitches that removes pitches where the map operation
    /// returns None, but retains those where the map operation returns Some(pitch).
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
    fn is_silent(&self) -> bool;
}

/// Macro to generate implementations of traits for pitches
macro_rules! impl_traits_for_pitches {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self = f(&self);
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self = f((None, &self));
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                if let Some(p) = f(&self) {
                    self = p;
                    Ok(self)
                } else {
                    Err(anyhow!(PitchError::RequiredPitchAbsent("Pitch.filter_map_pitch()".to_string())))
                }
            }

            fn transpose_pitch(mut self, n: $ty) -> Self {
                self += n;
                self
            }

            fn invert_pitch(mut self, n: $ty) -> Self {
                self = n + n - self;
                self
            }

            fn augment_pitch<MT: AugDim<$ty> + Copy>(mut self, n: MT) -> Self {
                n.augment_target(&mut self);
                self
            }

            fn diminish_pitch<MT: AugDim<$ty> + Copy>(mut self, n: MT) -> Self {
                n.diminish_target(&mut self);
                self
            }

            fn modulus(mut self, n: $ty) -> Self {
                // Can't use n.abs() as it's not implemented for unsigned ints
                let n = if n < $ty::zero() { $ty::zero() - n } else { n };
                let ret = self % n;

                self = if ret < $ty::zero() { ret + n } else { ret };
                self
            }

            fn trim_min(mut self, n: $ty) -> Self {
                if self < n {
                    self = n;
                }
                self
            }

            fn trim_max(mut self, n: $ty) -> Self {
                if self > n {
                    self = n;
                }
                self
            }

            fn trim(mut self, first: $ty, second: $ty) -> Self {
                let (min, max) = if first > second { (second, first ) } else { (first, second) };

                if self < min {
                    self = min;
                } else if self > max {
                    self = max;
                }
                self
            }

            fn is_silent(&self) -> bool {
                false
            }

            fn bounce_min(mut self, n: $ty) -> Self {
                if self < n {
                    self = n + n - self;
                }
                self
            }

            fn bounce_max(mut self, n: $ty) -> Self {
                if self > n {
                    self = n + n - self;
                }
                self
            }

            fn bounce(mut self, first: $ty, second: $ty) -> Self {
                let (min, max) = if first > second { (second, first ) } else { (first, second) };

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
    )*)
}

macro_rules! impl_fns_for_option {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(self, n: $ty) -> Self {
            self.map(|v| v.$fn(n))
        }
    )*)
}

macro_rules! impl_fns_for_vec {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.iter_mut().for_each(|p| { *p = p.$fn(n); });
            self
        }
    )*)
}

macro_rules! impl_fns_for_melody_member {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.values.iter_mut().for_each(|p| { *p = p.$fn(n); });
            self
        }
    )*)
}

macro_rules! impl_fns_for_seq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { *p = p.$fn(n); });
            self
        }
    )*)
}

macro_rules! impl_fns_for_chordseq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { p.iter_mut().for_each(|v| { *v = v.$fn(n); }); });
            self
        }
    )*)
}

macro_rules! impl_fns_for_melody {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { p.values.iter_mut().for_each(|v| { *v = v.$fn(n); }); });
            self
        }
    )*)
}

/// Macro to implement traits for entities containing pitches
macro_rules! impl_traits_for_pitch_containers {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for Option<$ty> {
            impl_fns_for_option!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(self, f: MapT) -> Self {
                self.map(|p| f(&p))
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(self, f: MapT) -> Self {
                self.map(|p| f((None, &p)))
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(self, f: MapT) -> Result<Self> {
                Ok(self.map(|p| f(&p)).flatten())
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(self, n: AT) -> Self {
                self.map(|p| p.augment_pitch(n))
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(self, n: AT) -> Self {
                self.map(|p| p.diminish_pitch(n))
            }

            fn trim(self, first: $ty, last: $ty) -> Self {
                self.map(|p| p.trim(first, last))
            }

            fn bounce(self, first: $ty, last: $ty) -> Self {
                self.map(|p| p.bounce(first, last))
            }

            fn is_silent(&self) -> bool {
                self.is_none()
            }
        }

        impl Pitch<$ty> for Vec<$ty> {
            impl_fns_for_vec!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.iter_mut().for_each(|p| *p = f(p));
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.iter_mut().for_each(|p| *p = f((None, p)));
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                self = self.into_iter().filter_map(|p| f(&p)).collect();
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.iter_mut().for_each(|p| { *p = p.augment_pitch(n); });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.iter_mut().for_each(|p| { *p = p.diminish_pitch(n); });
                self
            }

            fn trim(mut self, first: $ty, last: $ty) -> Self {
                self = self.into_iter().map(|p| p.trim(first, last)).collect();
                self
            }

            fn bounce(mut self, first: $ty, last: $ty) -> Self {
                self = self.into_iter().map(|p| p.bounce(first, last)).collect();
                self
            }

            fn is_silent(&self) -> bool {
                self.is_empty()
            }
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            impl_fns_for_melody_member!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.values.iter_mut().for_each(|p| *p = f(p));
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.values.iter_mut().for_each(|p| *p = f((None, p)));
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                self.values = self.values.into_iter().filter_map(|p| f(&p)).collect();
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.values.iter_mut().for_each(|p| { *p = p.augment_pitch(n); });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.values.iter_mut().for_each(|p| { *p = p.diminish_pitch(n); });
                self
            }

            fn trim(mut self, first: $ty, last: $ty) -> Self {
                self.values = self.values.trim(first, last);
                self
            }

            fn bounce(mut self, first: $ty, last: $ty) -> Self {
                self.values = self.values.bounce(first, last);
                self
            }

            fn is_silent(&self) -> bool {
                self.values.is_empty() || self.volume == 0
            }
        }

        impl Pitch<$ty> for NumericSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = f(p);
                });
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().enumerate().for_each(|(i, p)| *p = f((Some(i), p)));
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                for p in self.contents.iter_mut() {
                    *p = p.filter_map_pitch(&f)?;
                }
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.augment_pitch(n);
                });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.diminish_pitch(n);
                });
                self
            }

            fn trim(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.trim(first, second);
                });
                self
            }

            fn bounce(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.bounce(first, second);
                });
                self
            }

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }

        impl Pitch<$ty> for NoteSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.map(|v| f(&v));
                });
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().enumerate().for_each(|(i, p)| {
                    *p = p.map(|v| f((Some(i), &v)));
                });
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                self.contents.iter_mut().for_each(|p| *p = p.map(|v| f(&v)).flatten());
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.augment_pitch(n);
                });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.diminish_pitch(n);
                });
                self
            }

            fn trim(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.trim(first, second);
                });
                self
            }

            fn bounce(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    *p = p.bounce(first, second);
                });
                self
            }

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }

        impl Pitch<$ty> for ChordSeq<$ty> {
            impl_fns_for_chordseq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.iter_mut().for_each(|v| {
                        *v = f(&v);
                    });
                });
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().enumerate().for_each(|(i, p)| {
                    p.iter_mut().for_each(|v| {
                        *v = f((Some(i), &v));
                    });
                });
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                self.contents.iter_mut().for_each(|p| *p = p.into_iter().filter_map(|p| f(&p)).collect());
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.iter_mut().for_each(|v| {
                        *v = v.augment_pitch(n);
                    });
                });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.iter_mut().for_each(|v| {
                        *v = v.diminish_pitch(n);
                    });
                });
                self
            }

            fn trim(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.iter_mut().for_each(|v| {
                        *v = v.trim(first, second);
                    });
                });
                self
            }

            fn bounce(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.iter_mut().for_each(|v| {
                        *v = v.bounce(first, second);
                    });
                });
                self
            }

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }

        impl Pitch<$ty> for Melody<$ty> {
            impl_fns_for_melody!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

            fn map_pitch<MapT: Fn(&$ty) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.values.iter_mut().for_each(|v| {
                        *v = f(&v);
                    });
                });
                self
            }

            fn map_pitch_enumerated<MapT: Fn((Option<usize>, &$ty)) -> $ty>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().enumerate().for_each(|(i, p)| {
                    p.values.iter_mut().for_each(|v| {
                        *v = f((Some(i), &v));
                    });
                });
                self
            }

            fn filter_map_pitch<MapT: Fn(&$ty) -> Option<$ty>>(mut self, f: MapT) -> Result<Self> {
                self.contents.iter_mut().for_each(|p| {
                    let v = &mut p.values;

                    *v = v.into_iter().filter_map(|p| f(&p)).collect();
                });
                Ok(self)
            }

            fn augment_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.values.iter_mut().for_each(|v| {
                        *v = v.augment_pitch(n);
                    });
                });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.values.iter_mut().for_each(|v| {
                        *v = v.diminish_pitch(n);
                    });
                });
                self
            }

            fn trim(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.values.iter_mut().for_each(|v| {
                        *v = v.trim(first, second);
                    });
                });
                self
            }

            fn bounce(mut self, first: $ty, second: $ty) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.values.iter_mut().for_each(|v| {
                        *v = v.bounce(first, second);
                    });
                });
                self
            }

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }
    )*)
}

impl_traits_for_pitches!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);
impl_traits_for_pitch_containers!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::traits::Collection;
    use crate::entities::timing::DurationalEventTiming;
    use crate::metadata::MetadataList;
    use crate::{chordseq, melody, noteseq, numseq};
    use assert_float_eq::expect_f64_near;

    #[test]
    fn map_pitch() {
        macro_rules! map_pitch_test {
            ($init:expr, $fn:expr, $ret:expr) => {
                assert_eq!($init.map_pitch($fn), $ret);
            };
        }

        map_pitch_test!(3, |v| v + 1, 4);
        map_pitch_test!(None, |v| v + 1, None);
        map_pitch_test!(Some(3), |v| v + 1, Some(4));
        map_pitch_test!(vec![3, 4, 5], |v| v + 1, vec![4, 5, 6]);
        map_pitch_test!(
            MelodyMember::from(vec![3.0, 4.0, 5.0]),
            |v| v / 2.0,
            MelodyMember::from(vec![1.5, 2.0, 2.5])
        );
        map_pitch_test!(numseq![3.0, 4.0, 5.0], |v| v / 2.0, numseq![1.5, 2.0, 2.5]);
        map_pitch_test!(
            noteseq![3.0, None, 4.0, 5.0],
            |v| v / 2.0,
            noteseq![1.5, None, 2.0, 2.5]
        );
        map_pitch_test!(
            chordseq![[3.0], [], [4.0, 5.0]],
            |v| v / 2.0,
            chordseq![[1.5], [], [2.0, 2.5]]
        );
        map_pitch_test!(
            melody![[3.0], [], [4.0, 5.0]],
            |v| v / 2.0,
            melody![[1.5], [], [2.0, 2.5]]
        );
    }

    #[test]
    fn map_pitch_enumerated() {
        macro_rules! map_pitch_enumerated_test {
            ($init:expr, $fn:expr, $ret:expr) => {
                assert_eq!($init.map_pitch_enumerated($fn), $ret);
            };
        }

        map_pitch_enumerated_test!(3, |(_, v)| v + 1, 4);
        map_pitch_enumerated_test!(None, |(_, v)| v + 1, None);
        map_pitch_enumerated_test!(Some(3), |(_, v)| v + 1, Some(4));
        map_pitch_enumerated_test!(vec![3, 4, 5], |(_, v)| v + 1, vec![4, 5, 6]);
        map_pitch_enumerated_test!(
            MelodyMember::from(vec![3.0, 4.0, 5.0]),
            |(_, v)| v / 2.0,
            MelodyMember::from(vec![1.5, 2.0, 2.5])
        );
        map_pitch_enumerated_test!(
            numseq![3.0, 4.5, 5.5],
            |(i, v)| v * (1.0 + i.unwrap() as f32),
            numseq![3.0, 9.0, 16.5]
        );
        map_pitch_enumerated_test!(
            noteseq![3.0, None, 4.5, 5.5],
            |(i, v)| v * (1.0 + i.unwrap() as f32),
            noteseq![3.0, None, 13.5, 22.0]
        );
        map_pitch_enumerated_test!(
            chordseq![[3.0], [], [4.5, 5.5]],
            |(i, v)| v * (1.0 + i.unwrap() as f32),
            chordseq![[3.0], [], [13.5, 16.5]]
        );
        map_pitch_enumerated_test!(
            melody![[3.0], [], [4.5, 5.5]],
            |(i, v)| v * (1.0 + i.unwrap() as f32),
            melody![[3.0], [], [13.5, 16.5]]
        );
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
        filter_map_pitch_test!(None, |v| if v % 2 == 0 { Some(v / 2) } else { None }, None);
        filter_map_pitch_test!(
            Some(5),
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            None
        );
        filter_map_pitch_test!(
            Some(6),
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            Some(3)
        );
        filter_map_pitch_test!(
            vec![5, 6, 7],
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            vec![3]
        );
        filter_map_pitch_test!(
            MelodyMember::from(vec![5, 6, 7]),
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            MelodyMember::from(vec![3])
        );
        assert!(numseq![5, 6, 7]
            .filter_map_pitch(|v| if v % 2 == 0 { Some(v / 2) } else { None })
            .is_err());
        filter_map_pitch_test!(
            numseq![4, 6, 8],
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            numseq![2, 3, 4]
        );
        filter_map_pitch_test!(
            noteseq![5, None, 6, 7],
            |v| if v % 2 == 0 { Some(v / 2) } else { None },
            noteseq![None, None, 3, None]
        );
        filter_map_pitch_test!(
            chordseq![[4.4, 5.6], [], [5.3], [9.7, 11.1]],
            |v| if *v < 6.5 { Some(*v + 5.5) } else { None },
            chordseq![[9.9, 11.1], [], [10.8], []]
        );
        filter_map_pitch_test!(
            chordseq![[4.4, 5.6], [], [5.3], [9.7, 11.1]],
            |v| if *v < 6.5 { Some(*v + 5.5) } else { None },
            chordseq![[9.9, 11.1], [], [10.8], []]
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
        transpose_pitch_test!(None::<i32>, 6, None);
        transpose_pitch_test!(Some(5), 6, Some(11));
        transpose_pitch_test!(vec![4, 5, 6], 6, vec![10, 11, 12]);
        transpose_pitch_test!(
            MelodyMember::from(vec![4, 5, 6]),
            6,
            MelodyMember::from(vec![10, 11, 12])
        );
        transpose_pitch_test!(numseq![4, 5, 6], 6, numseq![10, 11, 12]);
        transpose_pitch_test!(noteseq![4, None, 6], 6, noteseq![10, None, 12]);
        transpose_pitch_test!(
            chordseq![[4, 5, 6], [], [7]],
            6,
            chordseq![[10, 11, 12], [], [13]]
        );
        transpose_pitch_test!(
            melody![[4, 5, 6], [], [7]],
            6,
            melody![[10, 11, 12], [], [13]]
        );
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
        invert_pitch_test!(None, 6, None);
        invert_pitch_test!(Some(5), 6, Some(7));
        invert_pitch_test!(vec![4, 5, 6], 6, vec![8, 7, 6]);
        invert_pitch_test!(
            MelodyMember::from(vec![4, 5, 6]),
            6,
            MelodyMember::from(vec![8, 7, 6])
        );
        invert_pitch_test!(numseq![4, 5, 6], 6, numseq![8, 7, 6]);
        invert_pitch_test!(noteseq![4, None, 6], 6, noteseq![8, None, 6]);
        invert_pitch_test!(
            chordseq![[4, 5, 6], [], [7]],
            6,
            chordseq![[8, 7, 6], [], [5]]
        );
        invert_pitch_test!(melody![[4, 5, 6], [], [7]], 6, melody![[8, 7, 6], [], [5]]);
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
        augment_pitch_test!(None::<i32>, 2, None);
        augment_pitch_test!(None::<i32>, 2.5, None);
        augment_pitch_test!(None::<f32>, 2, None);
        augment_pitch_test!(None::<f32>, 2.5, None);
        augment_pitch_test!(Some(4), 2, Some(8));
        augment_pitch_test!(Some(4), 2.5, Some(10));
        augment_pitch_test!(Some(2.5), 3, Some(7.5));
        augment_pitch_test!(Some(2.5), 3.0, Some(7.5));
        augment_pitch_test!(vec![4, 5, 7], 2, vec![8, 10, 14]);
        augment_pitch_test!(vec![4, 5, 7], 2.5, vec![10, 12, 17]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], 3, vec![12.0, 16.5, 19.5]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], 3.0, vec![12.0, 16.5, 19.5]);
        augment_pitch_test!(
            MelodyMember::from(vec![4, 5, 7]),
            2,
            MelodyMember::from(vec![8, 10, 14])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4, 5, 7]),
            2.5,
            MelodyMember::from(vec![10, 12, 17])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            3,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            3.0,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        augment_pitch_test!(numseq![4, 5, 7], 2, numseq![8, 10, 14]);
        augment_pitch_test!(numseq![4, 5, 7], 2.5, numseq![10, 12, 17]);
        augment_pitch_test!(numseq![4.0, 5.5, 6.5], 3, numseq![12.0, 16.5, 19.5]);
        augment_pitch_test!(numseq![4.0, 5.5, 6.5], 3.0, numseq![12.0, 16.5, 19.5]);
        augment_pitch_test!(noteseq![4, None, 5, 7], 2, noteseq![8, None, 10, 14]);
        augment_pitch_test!(noteseq![4, None, 5, 7], 2.5, noteseq![10, None, 12, 17]);
        augment_pitch_test!(
            noteseq![4.0, None, 5.5, 6.5],
            3,
            noteseq![12.0, None, 16.5, 19.5]
        );
        augment_pitch_test!(
            noteseq![4.0, None, 5.5, 6.5],
            3.0,
            noteseq![12.0, None, 16.5, 19.5]
        );
        augment_pitch_test!(chordseq![[4], [], [5, 7]], 2, chordseq![[8], [], [10, 14]]);
        augment_pitch_test!(
            chordseq![[4], [], [5, 7]],
            2.5,
            chordseq![[10], [], [12, 17]]
        );
        augment_pitch_test!(
            chordseq![[4.0], [], [5.5, 6.5]],
            3,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(
            chordseq![[4.0], [], [5.5, 6.5]],
            3.0,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(melody![[4], [], [5, 7]], 2, melody![[8], [], [10, 14]]);
        augment_pitch_test!(melody![[4], [], [5, 7]], 2.5, melody![[10], [], [12, 17]]);
        augment_pitch_test!(
            melody![[4.0], [], [5.5, 6.5]],
            3,
            melody![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(
            melody![[4.0], [], [5.5, 6.5]],
            3.0,
            melody![[12.0], [], [16.5, 19.5]]
        );
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
        diminish_pitch_test!(None::<i32>, 2, None);
        diminish_pitch_test!(None::<i32>, 2.5, None);
        diminish_pitch_test!(None::<f32>, 2, None);
        diminish_pitch_test!(None::<f32>, 2.5, None);
        diminish_pitch_test!(Some(8), 2, Some(4));
        diminish_pitch_test!(Some(10), 2.5, Some(4));
        diminish_pitch_test!(Some(7.5), 3, Some(2.5));
        diminish_pitch_test!(Some(7.5), 3.0, Some(2.5));
        diminish_pitch_test!(vec![8, 10, 14], 2, vec![4, 5, 7]);
        diminish_pitch_test!(vec![10, 12, 17], 2.5, vec![4, 4, 6]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], 3, vec![4.0, 5.5, 6.5]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], 3.0, vec![4.0, 5.5, 6.5]);
        diminish_pitch_test!(
            MelodyMember::from(vec![8, 10, 14]),
            2,
            MelodyMember::from(vec![4, 5, 7])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![10, 12, 17]),
            2.5,
            MelodyMember::from(vec![4, 4, 6])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            3,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            3.0,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        diminish_pitch_test!(numseq![8, 10, 14], 2, numseq![4, 5, 7]);
        diminish_pitch_test!(numseq![10, 12, 17], 2.5, numseq![4, 4, 6]);
        diminish_pitch_test!(numseq![12.0, 16.5, 19.5], 3, numseq![4.0, 5.5, 6.5]);
        diminish_pitch_test!(numseq![12.0, 16.5, 19.5], 3.0, numseq![4.0, 5.5, 6.5]);
        diminish_pitch_test!(noteseq![8, None, 10, 14], 2, noteseq![4, None, 5, 7]);
        diminish_pitch_test!(noteseq![10, None, 12, 17], 2.5, noteseq![4, None, 4, 6]);
        diminish_pitch_test!(
            noteseq![12.0, None, 16.5, 19.5],
            3,
            noteseq![4.0, None, 5.5, 6.5]
        );
        diminish_pitch_test!(
            noteseq![12.0, None, 16.5, 19.5],
            3.0,
            noteseq![4.0, None, 5.5, 6.5]
        );
        diminish_pitch_test!(chordseq![[8], [], [10, 14]], 2, chordseq![[4], [], [5, 7]]);
        diminish_pitch_test!(
            chordseq![[10], [], [12, 17]],
            2.5,
            chordseq![[4], [], [4, 6]]
        );
        diminish_pitch_test!(
            chordseq![[12.0], [], [16.5, 19.5]],
            3,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(
            chordseq![[12.0], [], [16.5, 19.5]],
            3.0,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(melody![[8], [], [10, 14]], 2, melody![[4], [], [5, 7]]);
        diminish_pitch_test!(melody![[10], [], [12, 17]], 2.5, melody![[4], [], [4, 6]]);
        diminish_pitch_test!(
            melody![[12.0], [], [16.5, 19.5]],
            3,
            melody![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(
            melody![[12.0], [], [16.5, 19.5]],
            3.0,
            melody![[4.0], [], [5.5, 6.5]]
        );
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
        modulus_test!(None, 3, None);
        modulus_test!(Some(5), 3, Some(2));
        modulus_test!(Some(-5), 3, Some(1));
        modulus_test!(vec![5.0, -5.0, 6.0], 1.5, vec![0.5, 1.0, 0.0]);
        modulus_test!(
            MelodyMember::from(vec![5, -5, 6]),
            3,
            MelodyMember::from(vec![2, 1, 0])
        );
        modulus_test!(numseq![5, -5, 6], 3, numseq![2, 1, 0]);
        modulus_test!(noteseq![5, None, -5, 6], 3, noteseq![2, None, 1, 0]);
        modulus_test!(chordseq![[5], [], [-5, 6]], 3, chordseq![[2], [], [1, 0]]);
        modulus_test!(
            melody![[5.0], [], [-5.0, 6.0]],
            1.5,
            melody![[0.5], [], [1.0, 0.0]]
        );
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
        trim_min_test!(None, 5, None);
        trim_min_test!(Some(5), 3, Some(5));
        trim_min_test!(Some(3), 5, Some(5));
        trim_min_test!(vec![5.0, 3.0], 4.0, vec![5.0, 4.0]);
        trim_min_test!(
            MelodyMember::from(vec![5.0, 3.0]),
            4.0,
            MelodyMember::from(vec![5.0, 4.0])
        );
        trim_min_test!(numseq![5.0, 3.0], 4.0, numseq![5.0, 4.0]);
        trim_min_test!(noteseq![5.0, None, 3.0], 4.0, noteseq![5.0, None, 4.0]);
        trim_min_test!(
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[5.0], [], [4.5, 4.0]]
        );
        trim_min_test!(
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[5.0], [], [4.5, 4.0]]
        );
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
        trim_max_test!(None, 5, None);
        trim_max_test!(Some(5), 3, Some(3));
        trim_max_test!(Some(3), 5, Some(3));
        trim_max_test!(vec![5.0, 3.0], 4.0, vec![4.0, 3.0]);
        trim_max_test!(
            MelodyMember::from(vec![5.0, 3.0]),
            4.0,
            MelodyMember::from(vec![4.0, 3.0])
        );
        trim_max_test!(numseq![5.0, 3.0], 4.0, numseq![4.0, 3.0]);
        trim_max_test!(noteseq![5.0, None, 3.0], 4.0, noteseq![4.0, None, 3.0]);
        trim_max_test!(
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[4.0], [], [4.0, 3.0]]
        );
        trim_max_test!(
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[4.0], [], [4.0, 3.0]]
        );
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
        trim_test!(None, 3, 5, None);
        trim_test!(Some(2), 3, 5, Some(3));
        trim_test!(Some(4), 3, 5, Some(4));
        trim_test!(Some(6), 3, 5, Some(5));
        trim_test!(vec![2, 4, 6], 3, 5, vec![3, 4, 5]);
        trim_test!(
            MelodyMember::from(vec![2.0, 4.0, 6.0]),
            5.0,
            3.0,
            MelodyMember::from(vec![3.0, 4.0, 5.0])
        );
        trim_test!(numseq![2.0, 4.0, 6.0], 3.0, 5.0, numseq![3.0, 4.0, 5.0]);
        trim_test!(
            noteseq![2.0, None, 4.0, 6.0],
            3.0,
            5.0,
            noteseq![3.0, None, 4.0, 5.0]
        );
        trim_test!(
            chordseq![[2.0], [], [4.0, 6.0]],
            5.0,
            3.0,
            chordseq![[3.0], [], [4.0, 5.0]]
        );
        trim_test!(
            melody![[2.0], [], [4.0, 6.0]],
            3.0,
            5.0,
            melody![[3.0], [], [4.0, 5.0]]
        );
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
        bounce_min_test!(None, 5, None);
        bounce_min_test!(Some(5), 3, Some(5));
        bounce_min_test!(Some(3), 5, Some(7));
        bounce_min_test!(vec![5.0, 3.0], 4.0, vec![5.0, 5.0]);
        bounce_min_test!(
            MelodyMember::from(vec![5.0, 3.0]),
            4.0,
            MelodyMember::from(vec![5.0, 5.0])
        );
        bounce_min_test!(numseq![5.0, 3.0], 4.0, numseq![5.0, 5.0]);
        bounce_min_test!(noteseq![5.0, None, 3.0], 4.0, noteseq![5.0, None, 5.0]);
        bounce_min_test!(
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[5.0], [], [4.5, 5.0]]
        );
        bounce_min_test!(
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[5.0], [], [4.5, 5.0]]
        );
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
        bounce_max_test!(None, 5, None);
        bounce_max_test!(Some(5), 3, Some(1));
        bounce_max_test!(Some(3), 5, Some(3));
        bounce_max_test!(vec![5.0, 3.0], 4.0, vec![3.0, 3.0]);
        bounce_max_test!(
            MelodyMember::from(vec![3.0, 5.0]),
            4.0,
            MelodyMember::from(vec![3.0, 3.0])
        );
        bounce_max_test!(numseq![5.0, 3.0], 4.0, numseq![3.0, 3.0]);
        bounce_max_test!(noteseq![5.0, None, 3.0], 4.0, noteseq![3.0, None, 3.0]);
        bounce_max_test!(
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[3.0], [], [3.5, 3.0]]
        );
        bounce_max_test!(
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[3.0], [], [3.5, 3.0]]
        );
    }

    #[test]
    fn bounce() {
        macro_rules! bounce_test {
            ($init:expr, $arg1:expr, $arg2:expr, $ret:expr) => {
                assert_eq!($init.bounce($arg1, $arg2), $ret);
            };
        }

        macro_rules! bounce_seq_test {
            ($init:expr, $arg1:expr, $arg2:expr, $ret:expr) => {
                let result = $init.bounce($arg1, $arg2);
                let expected = $ret;

                result
                    .contents
                    .into_iter()
                    .zip(expected.contents.into_iter())
                    .enumerate()
                    .for_each(|(i, (ret, exp))| {
                        assert!(
                            expect_f64_near!(ret, exp).is_ok(),
                            "index {i} failed: expected {ret}, got {exp}"
                        );
                    });
            };
        }

        bounce_test!(2, 3, 6, 4);
        bounce_test!(4, 3, 6, 4);
        bounce_test!(8, 3, 6, 4);
        bounce_test!(-1, 3, 6, 5);
        bounce_test!(11, 3, 6, 5);
        bounce_test!(11, 6, 6, 6);
        bounce_test!(None, 6, 3, None);
        bounce_test!(Some(2), 6, 3, Some(4));
        bounce_test!(Some(4), 6, 3, Some(4));
        bounce_test!(Some(8), 6, 3, Some(4));
        bounce_test!(Some(8), 6, 6, Some(6));
        bounce_test!(vec![2, 4, 8, -1, 11], 3, 6, vec![4, 4, 4, 5, 5]);
        bounce_test!(vec![2, 4, 8, -1, 11], 6, 6, vec![6, 6, 6, 6, 6]);
        bounce_test!(
            MelodyMember::from(vec![2, 4, 8, -1, 11]),
            3,
            6,
            MelodyMember::from(vec![4, 4, 4, 5, 5])
        );
        bounce_seq_test!(
            numseq![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            4.2,
            5.6,
            numseq![4.6, 4.8, 5.4, 4.4, 5.0, 5.2, 4.2, 5.2, 5.0, 4.4]
        );
        bounce_test!(noteseq![2, None, 4, 8], 3, 6, noteseq![4, None, 4, 4]);
        bounce_test!(chordseq![[2], [], [4, 8]], 3, 6, chordseq![[4], [], [4, 4]]);
        bounce_test!(melody![[2], [], [4, 8]], 3, 6, melody![[4], [], [4, 4]]);
        bounce_test!(melody![[2], [], [4, 8]], 6, 6, melody![[6], [], [6, 6]]);
    }

    #[test]
    fn is_silent() {
        assert!(!0.is_silent());
        assert!(!Some(0).is_silent());
        assert!(None::<i32>.is_silent());
        assert!(Vec::<i32>::new().is_silent());
        assert!(!vec![0].is_silent());
        assert!(MelodyMember::<i32>::from(vec![]).is_silent());
        assert!(!MelodyMember::from(vec![0]).is_silent());
        assert!(MelodyMember {
            values: vec![0],
            volume: 0,
            timing: DurationalEventTiming::default(),
            before: MetadataList::default()
        }
        .is_silent());
        assert!(NumericSeq::<i32>::new(vec![]).is_silent());
        assert!(!numseq![0].is_silent());
        assert!(NoteSeq::<i32>::new(vec![]).is_silent());
        assert!(NoteSeq::<i32>::new(vec![None]).is_silent());
        assert!(!noteseq![0, None, 1].is_silent());
        assert!(ChordSeq::<i32>::new(vec![]).is_silent());
        assert!(ChordSeq::<i32>::new(vec![vec![]]).is_silent());
        assert!(!chordseq![[0], [], [1]].is_silent());
        assert!(Melody::<i32>::new(vec![]).is_silent());
        assert!(Melody::<i32>::try_from(vec![vec![]]).unwrap().is_silent());
        assert!(!melody![[0], [], [1]].is_silent());
        assert!(Melody::new(vec![MelodyMember {
            values: vec![0],
            volume: 0,
            timing: DurationalEventTiming::default(),
            before: MetadataList::default()
        }])
        .is_silent());
    }
}
