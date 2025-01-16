use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
};
use num_traits::Zero;

pub trait AugDim<MT> {
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

/// A trait that implements various operations that can be executed on pitches
/// or containers for pitches.
pub trait Pitch<T> {
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
    fn augment_pitch<AT: AugDim<T>>(self, n: &AT) -> Self;

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
    fn diminish_pitch<AT: AugDim<T>>(self, n: &AT) -> Self;

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

    /// Is this pitch or pitch container silent?
    fn is_silent(&self) -> bool;
}

macro_rules! impl_fns_for_option {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            if let Some(p) = self.as_mut() {
                self = Some(p.$fn(n));
            }
            self
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

macro_rules! impl_other_fns_for_seq {
    ($ty:ident) => {
        fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
            self.contents.iter_mut().for_each(|p| {
                *p = p.augment_pitch(n);
            });
            self
        }

        fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
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

        fn is_silent(&self) -> bool {
            self.contents.iter().all(|m| m.is_silent())
        }
    };
}

macro_rules! impl_fns_for_chordseq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { p.iter_mut().for_each(|v| { *v = v.$fn(n); }); });
            self
        }
    )*)
}

macro_rules! impl_other_fns_for_chordseq {
    ($ty:ident) => {
        fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
            self.contents.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|v| {
                    *v = v.augment_pitch(n);
                });
            });
            self
        }

        fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
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

        fn is_silent(&self) -> bool {
            self.contents.iter().all(|m| m.is_silent())
        }
    };
}

macro_rules! impl_fns_for_melody {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| { p.values.iter_mut().for_each(|v| { *v = v.$fn(n); }); });
            self
        }
    )*)
}

macro_rules! impl_other_fns_for_melody {
    ($ty:ident) => {
        fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
            self.contents.iter_mut().for_each(|p| {
                p.values.iter_mut().for_each(|v| {
                    *v = v.augment_pitch(n);
                });
            });
            self
        }

        fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
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

        fn is_silent(&self) -> bool {
            self.contents.iter().all(|m| m.is_silent())
        }
    };
}

macro_rules! impl_traits_for_raw_values {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn transpose_pitch(mut self, n: $ty) -> Self {
                self += n;
                self
            }

            fn invert_pitch(mut self, n: $ty) -> Self {
                self = n + n - self;
                self
            }

            fn augment_pitch<MT: AugDim<$ty>>(mut self, n: &MT) -> Self {
                n.augment_target(&mut self);
                self
            }

            fn diminish_pitch<MT: AugDim<$ty>>(mut self, n: &MT) -> Self {
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
                let (low, hi) = if first > second { (second, first ) } else { (first, second) };

                if self < low {
                    self = low;
                } else if self > hi {
                    self = hi;
                }

                self
            }

            fn is_silent(&self) -> bool {
                false
            }
        }
    )*)
}

macro_rules! impl_traits_for_derived_entities {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for Option<$ty> {
            impl_fns_for_option!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                if let Some(p) = self.as_mut() {
                    self = Some(p.augment_pitch(n));
                }
                self
            }

            fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                if let Some(p) = self.as_mut() {
                    self = Some(p.diminish_pitch(n));
                }
                self
            }

            fn trim(mut self, first: $ty, last: $ty) -> Self {
                if let Some(p) = self.as_mut() {
                    self = Some(p.trim(first, last));
                }
                self
            }

            fn is_silent(&self) -> bool {
                self.is_none()
            }
        }

        impl Pitch<$ty> for Vec<$ty> {
            impl_fns_for_vec!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                self.iter_mut().for_each(|p| { *p = p.augment_pitch(n); });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                self.iter_mut().for_each(|p| { *p = p.diminish_pitch(n); });
                self
            }

            fn trim(mut self, first: $ty, last: $ty) -> Self {
                self = self.into_iter().map(|p| p.trim(first, last)).collect();
                self
            }

            fn is_silent(&self) -> bool {
                self.is_empty()
            }
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            impl_fns_for_melody_member!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                self.values.iter_mut().for_each(|p| { *p = p.augment_pitch(n); });
                self
            }

            fn diminish_pitch<AT: AugDim<$ty>>(mut self, n: &AT) -> Self {
                self.values.iter_mut().for_each(|p| { *p = p.diminish_pitch(n); });
                self
            }

            fn trim(mut self, first: $ty, last: $ty) -> Self {
                self.values = self.values.trim(first, last);
                self
            }

            fn is_silent(&self) -> bool {
                self.values.is_empty() || self.volume == 0
            }
        }

        impl Pitch<$ty> for NumericSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);
            impl_other_fns_for_seq!($ty);
        }

        impl Pitch<$ty> for NoteSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);
            impl_other_fns_for_seq!($ty);
        }

        impl Pitch<$ty> for ChordSeq<$ty> {
            impl_fns_for_chordseq!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);
            impl_other_fns_for_chordseq!($ty);
        }

        impl Pitch<$ty> for Melody<$ty> {
            impl_fns_for_melody!($ty, for transpose_pitch invert_pitch modulus trim_min trim_max);
            impl_other_fns_for_melody!($ty);
        }
    )*)
}

impl_traits_for_raw_values!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);
impl_traits_for_derived_entities!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::traits::Collection;
    use crate::entities::timing::DurationalEventTiming;
    use crate::metadata::MetadataList;
    use crate::{chordseq, melody, noteseq, numseq};

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

        augment_pitch_test!(4, &2, 8);
        augment_pitch_test!(4, &2.5, 10);
        augment_pitch_test!(2.5, &3, 7.5);
        augment_pitch_test!(2.5, &3.0, 7.5);
        augment_pitch_test!(None::<i32>, &2, None);
        augment_pitch_test!(None::<i32>, &2.5, None);
        augment_pitch_test!(None::<f32>, &2, None);
        augment_pitch_test!(None::<f32>, &2.5, None);
        augment_pitch_test!(Some(4), &2, Some(8));
        augment_pitch_test!(Some(4), &2.5, Some(10));
        augment_pitch_test!(Some(2.5), &3, Some(7.5));
        augment_pitch_test!(Some(2.5), &3.0, Some(7.5));
        augment_pitch_test!(vec![4, 5, 7], &2, vec![8, 10, 14]);
        augment_pitch_test!(vec![4, 5, 7], &2.5, vec![10, 12, 17]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], &3, vec![12.0, 16.5, 19.5]);
        augment_pitch_test!(vec![4.0, 5.5, 6.5], &3.0, vec![12.0, 16.5, 19.5]);
        augment_pitch_test!(
            MelodyMember::from(vec![4, 5, 7]),
            &2,
            MelodyMember::from(vec![8, 10, 14])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4, 5, 7]),
            &2.5,
            MelodyMember::from(vec![10, 12, 17])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            &3,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        augment_pitch_test!(
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            &3.0,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        augment_pitch_test!(numseq![4, 5, 7], &2, numseq![8, 10, 14]);
        augment_pitch_test!(numseq![4, 5, 7], &2.5, numseq![10, 12, 17]);
        augment_pitch_test!(numseq![4.0, 5.5, 6.5], &3, numseq![12.0, 16.5, 19.5]);
        augment_pitch_test!(numseq![4.0, 5.5, 6.5], &3.0, numseq![12.0, 16.5, 19.5]);
        augment_pitch_test!(noteseq![4, None, 5, 7], &2, noteseq![8, None, 10, 14]);
        augment_pitch_test!(noteseq![4, None, 5, 7], &2.5, noteseq![10, None, 12, 17]);
        augment_pitch_test!(
            noteseq![4.0, None, 5.5, 6.5],
            &3,
            noteseq![12.0, None, 16.5, 19.5]
        );
        augment_pitch_test!(
            noteseq![4.0, None, 5.5, 6.5],
            &3.0,
            noteseq![12.0, None, 16.5, 19.5]
        );
        augment_pitch_test!(chordseq![[4], [], [5, 7]], &2, chordseq![[8], [], [10, 14]]);
        augment_pitch_test!(
            chordseq![[4], [], [5, 7]],
            &2.5,
            chordseq![[10], [], [12, 17]]
        );
        augment_pitch_test!(
            chordseq![[4.0], [], [5.5, 6.5]],
            &3,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(
            chordseq![[4.0], [], [5.5, 6.5]],
            &3.0,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(melody![[4], [], [5, 7]], &2, melody![[8], [], [10, 14]]);
        augment_pitch_test!(melody![[4], [], [5, 7]], &2.5, melody![[10], [], [12, 17]]);
        augment_pitch_test!(
            melody![[4.0], [], [5.5, 6.5]],
            &3,
            melody![[12.0], [], [16.5, 19.5]]
        );
        augment_pitch_test!(
            melody![[4.0], [], [5.5, 6.5]],
            &3.0,
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

        diminish_pitch_test!(8, &2, 4);
        diminish_pitch_test!(10, &2.5, 4);
        diminish_pitch_test!(7.5, &3, 2.5);
        diminish_pitch_test!(7.5, &3.0, 2.5);
        diminish_pitch_test!(None::<i32>, &2, None);
        diminish_pitch_test!(None::<i32>, &2.5, None);
        diminish_pitch_test!(None::<f32>, &2, None);
        diminish_pitch_test!(None::<f32>, &2.5, None);
        diminish_pitch_test!(Some(8), &2, Some(4));
        diminish_pitch_test!(Some(10), &2.5, Some(4));
        diminish_pitch_test!(Some(7.5), &3, Some(2.5));
        diminish_pitch_test!(Some(7.5), &3.0, Some(2.5));
        diminish_pitch_test!(vec![8, 10, 14], &2, vec![4, 5, 7]);
        diminish_pitch_test!(vec![10, 12, 17], &2.5, vec![4, 4, 6]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], &3, vec![4.0, 5.5, 6.5]);
        diminish_pitch_test!(vec![12.0, 16.5, 19.5], &3.0, vec![4.0, 5.5, 6.5]);
        diminish_pitch_test!(
            MelodyMember::from(vec![8, 10, 14]),
            &2,
            MelodyMember::from(vec![4, 5, 7])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![10, 12, 17]),
            &2.5,
            MelodyMember::from(vec![4, 4, 6])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            &3,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        diminish_pitch_test!(
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            &3.0,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        diminish_pitch_test!(numseq![8, 10, 14], &2, numseq![4, 5, 7]);
        diminish_pitch_test!(numseq![10, 12, 17], &2.5, numseq![4, 4, 6]);
        diminish_pitch_test!(numseq![12.0, 16.5, 19.5], &3, numseq![4.0, 5.5, 6.5]);
        diminish_pitch_test!(numseq![12.0, 16.5, 19.5], &3.0, numseq![4.0, 5.5, 6.5]);
        diminish_pitch_test!(noteseq![8, None, 10, 14], &2, noteseq![4, None, 5, 7]);
        diminish_pitch_test!(noteseq![10, None, 12, 17], &2.5, noteseq![4, None, 4, 6]);
        diminish_pitch_test!(
            noteseq![12.0, None, 16.5, 19.5],
            &3,
            noteseq![4.0, None, 5.5, 6.5]
        );
        diminish_pitch_test!(
            noteseq![12.0, None, 16.5, 19.5],
            &3.0,
            noteseq![4.0, None, 5.5, 6.5]
        );
        diminish_pitch_test!(chordseq![[8], [], [10, 14]], &2, chordseq![[4], [], [5, 7]]);
        diminish_pitch_test!(
            chordseq![[10], [], [12, 17]],
            &2.5,
            chordseq![[4], [], [4, 6]]
        );
        diminish_pitch_test!(
            chordseq![[12.0], [], [16.5, 19.5]],
            &3,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(
            chordseq![[12.0], [], [16.5, 19.5]],
            &3.0,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(melody![[8], [], [10, 14]], &2, melody![[4], [], [5, 7]]);
        diminish_pitch_test!(melody![[10], [], [12, 17]], &2.5, melody![[4], [], [4, 6]]);
        diminish_pitch_test!(
            melody![[12.0], [], [16.5, 19.5]],
            &3,
            melody![[4.0], [], [5.5, 6.5]]
        );
        diminish_pitch_test!(
            melody![[12.0], [], [16.5, 19.5]],
            &3.0,
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
