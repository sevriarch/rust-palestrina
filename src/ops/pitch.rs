use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
};

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
    fn transpose_pitch(&mut self, n: T);

    /// Invert pitches around a specific value.
    fn invert_pitch(&mut self, n: T);

    /// Multiply pitches by a specific value.
    ///
    /// This value does not have to be the same type as the pitches being multiplied
    /// (automatic conversion will be performed) and it can be an integer or a float.
    ///
    /// Note that in the case where pitches are of an integer type and the multiplier
    /// is of a floating point type, results will always be rounded down.
    fn augment_pitch<AT: AugDim<T>>(&mut self, n: &AT);

    /// Divide pitches by a specific value.
    ///
    /// This value does not have to be the same type as the pitches being divided
    /// (automatic conversion will be performed) and it can be an integer or a float.
    ///
    /// Note that in the case where pitches are of an integer type and the divisor
    /// is of a floating point type, results will always be rounded down.
    fn diminish_pitch<AT: AugDim<T>>(&mut self, n: &AT);

    /// Replace any pitches below the passed argument with the passed argument.
    fn trim_min(&mut self, n: T);

    /// Replace any pitches above the passed argument with the passed argument.
    fn trim_max(&mut self, n: T);

    /// Replace any pitches below the lower argument with the lower argument.
    /// Replace any pitches above the higher argument with the higher argument.
    fn trim(&mut self, first: T, second: T);

    /// Is this pitch or pitch container silent?
    fn is_silent(&self) -> bool;
}

macro_rules! impl_fns_for_option {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(&mut self, n: $ty) {
            if let Some(p) = self.as_mut() {
                p.$fn(n);
            }
        }
    )*)
}

macro_rules! impl_fns_for_vec {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(&mut self, n: $ty) {
            self.iter_mut().for_each(|p| p.$fn(n));
        }
    )*)
}

macro_rules! impl_fns_for_melody_member {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(&mut self, n: $ty) {
            self.values.$fn(n)
        }
    )*)
}

macro_rules! impl_fns_for_seq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(&mut self, n: $ty) {
            self.contents.iter_mut().for_each(|p| p.$fn(n));
        }
    )*)
}

macro_rules! impl_other_fns_for_seq {
    ($ty:ident) => {
        fn augment_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
            self.contents.iter_mut().for_each(|p| p.augment_pitch(n));
        }

        fn diminish_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
            self.contents.iter_mut().for_each(|p| p.diminish_pitch(n));
        }

        fn trim(&mut self, first: $ty, second: $ty) {
            self.contents.iter_mut().for_each(|p| p.trim(first, second));
        }

        fn is_silent(&self) -> bool {
            self.contents.iter().all(|m| m.is_silent())
        }
    };
}

macro_rules! impl_traits_for_raw_values {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn transpose_pitch(&mut self, n: $ty) {
                *self += n
            }

            fn invert_pitch(&mut self, n: $ty) {
                *self = n + n - *self
            }

            fn augment_pitch<MT: AugDim<$ty>>(&mut self, n: &MT) {
                n.augment_target(self)
            }

            fn diminish_pitch<MT: AugDim<$ty>>(&mut self, n: &MT) {
                n.diminish_target(self)
            }

            fn is_silent(&self) -> bool {
                false
            }

            fn trim_min(&mut self, n: $ty) {
                if *self < n {
                    *self = n;
                }
            }

            fn trim_max(&mut self, n: $ty) {
                if *self > n {
                    *self = n;
                }
            }

            fn trim(&mut self, first: $ty, second: $ty) {
                let (low, hi) = if first > second { (second, first ) } else { (first, second) };

                if *self < low {
                    *self = low;
                } else if *self > hi {
                    *self = hi;
                }
            }
        }
    )*)
}

macro_rules! impl_traits_for_derived_entities {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for Option<$ty> {
            impl_fns_for_option!($ty, for transpose_pitch invert_pitch trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                if let Some(p) = self.as_mut() {
                    p.augment_pitch(n);
                }
            }

            fn diminish_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                if let Some(p) = self.as_mut() {
                    p.diminish_pitch(n);
                }
            }

            fn trim(&mut self, first: $ty, last: $ty) {
                if let Some(p) = self.as_mut() {
                    p.trim(first, last);
                }
            }

            fn is_silent(&self) -> bool {
                self.is_none()
            }
        }

        impl Pitch<$ty> for Vec<$ty> {
            impl_fns_for_vec!($ty, for transpose_pitch invert_pitch trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                self.iter_mut().for_each(|p| p.augment_pitch(n));
            }

            fn diminish_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                self.iter_mut().for_each(|p| p.diminish_pitch(n));
            }

            fn trim(&mut self, first: $ty, last: $ty) {
                self.iter_mut().for_each(|p| p.trim(first, last));
            }

            fn is_silent(&self) -> bool {
                self.is_empty()
            }
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            impl_fns_for_melody_member!($ty, for transpose_pitch invert_pitch trim_min trim_max);

            fn augment_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                self.values.augment_pitch(n);
            }

            fn diminish_pitch<AT: AugDim<$ty>>(&mut self, n: &AT) {
                self.values.diminish_pitch(n);
            }

            fn trim(&mut self, first: $ty, last: $ty) {
                self.values.trim(first, last);
            }

            fn is_silent(&self) -> bool {
                self.values.is_empty() || self.volume == 0
            }
        }

        impl Pitch<$ty> for NumericSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch trim_min trim_max);
            impl_other_fns_for_seq!($ty);
        }

        impl Pitch<$ty> for NoteSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch trim_min trim_max);
            impl_other_fns_for_seq!($ty);
        }

        impl Pitch<$ty> for ChordSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch trim_min trim_max);
            impl_other_fns_for_seq!($ty);
        }

        impl Pitch<$ty> for Melody<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch trim_min trim_max);
            impl_other_fns_for_seq!($ty);
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

    macro_rules! pitch_trait_test {
        ($fn:ident, $init:expr, $arg:expr, $ret:expr) => {
            let mut x = $init;
            x.$fn($arg);
            assert_eq!(x, $ret);
        };

        ($fn:ident, $init:expr, $arg1:expr, $arg2:expr, $ret:expr) => {
            let mut x = $init;
            x.$fn($arg1, $arg2);
            assert_eq!(x, $ret);
        };
    }

    #[test]
    fn transpose_pitch() {
        pitch_trait_test!(transpose_pitch, 5, 6, 11);
        pitch_trait_test!(transpose_pitch, 5.5, 6.0, 11.5);
        pitch_trait_test!(transpose_pitch, None, 6, None);
        pitch_trait_test!(transpose_pitch, Some(5), 6, Some(11));
        pitch_trait_test!(transpose_pitch, vec![4, 5, 6], 6, vec![10, 11, 12]);
        pitch_trait_test!(
            transpose_pitch,
            MelodyMember::from(vec![4, 5, 6]),
            6,
            MelodyMember::from(vec![10, 11, 12])
        );
        pitch_trait_test!(transpose_pitch, numseq![4, 5, 6], 6, numseq![10, 11, 12]);
        pitch_trait_test!(
            transpose_pitch,
            noteseq![4, None, 6],
            6,
            noteseq![10, None, 12]
        );
        pitch_trait_test!(
            transpose_pitch,
            chordseq![[4, 5, 6], [], [7]],
            6,
            chordseq![[10, 11, 12], [], [13]]
        );
        pitch_trait_test!(
            transpose_pitch,
            melody![[4, 5, 6], [], [7]],
            6,
            melody![[10, 11, 12], [], [13]]
        );
    }

    #[test]
    fn invert_pitch() {
        pitch_trait_test!(invert_pitch, 5, 6, 7);
        pitch_trait_test!(invert_pitch, 5.5, 6.0, 6.5);
        pitch_trait_test!(invert_pitch, None, 6, None);
        pitch_trait_test!(invert_pitch, Some(5), 6, Some(7));
        pitch_trait_test!(invert_pitch, vec![4, 5, 6], 6, vec![8, 7, 6]);
        pitch_trait_test!(
            invert_pitch,
            MelodyMember::from(vec![4, 5, 6]),
            6,
            MelodyMember::from(vec![8, 7, 6])
        );
        pitch_trait_test!(invert_pitch, numseq![4, 5, 6], 6, numseq![8, 7, 6]);
        pitch_trait_test!(invert_pitch, noteseq![4, None, 6], 6, noteseq![8, None, 6]);
        pitch_trait_test!(
            invert_pitch,
            chordseq![[4, 5, 6], [], [7]],
            6,
            chordseq![[8, 7, 6], [], [5]]
        );
        pitch_trait_test!(
            invert_pitch,
            melody![[4, 5, 6], [], [7]],
            6,
            melody![[8, 7, 6], [], [5]]
        );
    }

    #[test]
    fn augment_pitch() {
        pitch_trait_test!(augment_pitch, 4, &2, 8);
        pitch_trait_test!(augment_pitch, 4, &2.5, 10);
        pitch_trait_test!(augment_pitch, 2.5, &3, 7.5);
        pitch_trait_test!(augment_pitch, 2.5, &3.0, 7.5);
        pitch_trait_test!(augment_pitch, None::<i32>, &2, None);
        pitch_trait_test!(augment_pitch, None::<i32>, &2.5, None);
        pitch_trait_test!(augment_pitch, None::<f32>, &2, None);
        pitch_trait_test!(augment_pitch, None::<f32>, &2.5, None);
        pitch_trait_test!(augment_pitch, Some(4), &2, Some(8));
        pitch_trait_test!(augment_pitch, Some(4), &2.5, Some(10));
        pitch_trait_test!(augment_pitch, Some(2.5), &3, Some(7.5));
        pitch_trait_test!(augment_pitch, Some(2.5), &3.0, Some(7.5));
        pitch_trait_test!(augment_pitch, vec![4, 5, 7], &2, vec![8, 10, 14]);
        pitch_trait_test!(augment_pitch, vec![4, 5, 7], &2.5, vec![10, 12, 17]);
        pitch_trait_test!(
            augment_pitch,
            vec![4.0, 5.5, 6.5],
            &3,
            vec![12.0, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            vec![4.0, 5.5, 6.5],
            &3.0,
            vec![12.0, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            MelodyMember::from(vec![4, 5, 7]),
            &2,
            MelodyMember::from(vec![8, 10, 14])
        );
        pitch_trait_test!(
            augment_pitch,
            MelodyMember::from(vec![4, 5, 7]),
            &2.5,
            MelodyMember::from(vec![10, 12, 17])
        );
        pitch_trait_test!(
            augment_pitch,
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            &3,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        pitch_trait_test!(
            augment_pitch,
            MelodyMember::from(vec![4.0, 5.5, 6.5]),
            &3.0,
            MelodyMember::from(vec![12.0, 16.5, 19.5])
        );
        pitch_trait_test!(augment_pitch, numseq![4, 5, 7], &2, numseq![8, 10, 14]);
        pitch_trait_test!(augment_pitch, numseq![4, 5, 7], &2.5, numseq![10, 12, 17]);
        pitch_trait_test!(
            augment_pitch,
            numseq![4.0, 5.5, 6.5],
            &3,
            numseq![12.0, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            numseq![4.0, 5.5, 6.5],
            &3.0,
            numseq![12.0, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            noteseq![4, None, 5, 7],
            &2,
            noteseq![8, None, 10, 14]
        );
        pitch_trait_test!(
            augment_pitch,
            noteseq![4, None, 5, 7],
            &2.5,
            noteseq![10, None, 12, 17]
        );
        pitch_trait_test!(
            augment_pitch,
            noteseq![4.0, None, 5.5, 6.5],
            &3,
            noteseq![12.0, None, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            noteseq![4.0, None, 5.5, 6.5],
            &3.0,
            noteseq![12.0, None, 16.5, 19.5]
        );
        pitch_trait_test!(
            augment_pitch,
            chordseq![[4], [], [5, 7]],
            &2,
            chordseq![[8], [], [10, 14]]
        );
        pitch_trait_test!(
            augment_pitch,
            chordseq![[4], [], [5, 7]],
            &2.5,
            chordseq![[10], [], [12, 17]]
        );
        pitch_trait_test!(
            augment_pitch,
            chordseq![[4.0], [], [5.5, 6.5]],
            &3,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        pitch_trait_test!(
            augment_pitch,
            chordseq![[4.0], [], [5.5, 6.5]],
            &3.0,
            chordseq![[12.0], [], [16.5, 19.5]]
        );
        pitch_trait_test!(
            augment_pitch,
            melody![[4], [], [5, 7]],
            &2,
            melody![[8], [], [10, 14]]
        );
        pitch_trait_test!(
            augment_pitch,
            melody![[4], [], [5, 7]],
            &2.5,
            melody![[10], [], [12, 17]]
        );
        pitch_trait_test!(
            augment_pitch,
            melody![[4.0], [], [5.5, 6.5]],
            &3,
            melody![[12.0], [], [16.5, 19.5]]
        );
        pitch_trait_test!(
            augment_pitch,
            melody![[4.0], [], [5.5, 6.5]],
            &3.0,
            melody![[12.0], [], [16.5, 19.5]]
        );
    }

    #[test]
    fn diminish_pitch() {
        pitch_trait_test!(diminish_pitch, 8, &2, 4);
        pitch_trait_test!(diminish_pitch, 10, &2.5, 4);
        pitch_trait_test!(diminish_pitch, 7.5, &3, 2.5);
        pitch_trait_test!(diminish_pitch, 7.5, &3.0, 2.5);
        pitch_trait_test!(diminish_pitch, None::<i32>, &2, None);
        pitch_trait_test!(diminish_pitch, None::<i32>, &2.5, None);
        pitch_trait_test!(diminish_pitch, None::<f32>, &2, None);
        pitch_trait_test!(diminish_pitch, None::<f32>, &2.5, None);
        pitch_trait_test!(diminish_pitch, Some(8), &2, Some(4));
        pitch_trait_test!(diminish_pitch, Some(10), &2.5, Some(4));
        pitch_trait_test!(diminish_pitch, Some(7.5), &3, Some(2.5));
        pitch_trait_test!(diminish_pitch, Some(7.5), &3.0, Some(2.5));
        pitch_trait_test!(diminish_pitch, vec![8, 10, 14], &2, vec![4, 5, 7]);
        pitch_trait_test!(diminish_pitch, vec![10, 12, 17], &2.5, vec![4, 4, 6]);
        pitch_trait_test!(
            diminish_pitch,
            vec![12.0, 16.5, 19.5],
            &3,
            vec![4.0, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            vec![12.0, 16.5, 19.5],
            &3.0,
            vec![4.0, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            MelodyMember::from(vec![8, 10, 14]),
            &2,
            MelodyMember::from(vec![4, 5, 7])
        );
        pitch_trait_test!(
            diminish_pitch,
            MelodyMember::from(vec![10, 12, 17]),
            &2.5,
            MelodyMember::from(vec![4, 4, 6])
        );
        pitch_trait_test!(
            diminish_pitch,
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            &3,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        pitch_trait_test!(
            diminish_pitch,
            MelodyMember::from(vec![12.0, 16.5, 19.5]),
            &3.0,
            MelodyMember::from(vec![4.0, 5.5, 6.5])
        );
        pitch_trait_test!(diminish_pitch, numseq![8, 10, 14], &2, numseq![4, 5, 7]);
        pitch_trait_test!(diminish_pitch, numseq![10, 12, 17], &2.5, numseq![4, 4, 6]);
        pitch_trait_test!(
            diminish_pitch,
            numseq![12.0, 16.5, 19.5],
            &3,
            numseq![4.0, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            numseq![12.0, 16.5, 19.5],
            &3.0,
            numseq![4.0, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            noteseq![8, None, 10, 14],
            &2,
            noteseq![4, None, 5, 7]
        );
        pitch_trait_test!(
            diminish_pitch,
            noteseq![10, None, 12, 17],
            &2.5,
            noteseq![4, None, 4, 6]
        );
        pitch_trait_test!(
            diminish_pitch,
            noteseq![12.0, None, 16.5, 19.5],
            &3,
            noteseq![4.0, None, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            noteseq![12.0, None, 16.5, 19.5],
            &3.0,
            noteseq![4.0, None, 5.5, 6.5]
        );
        pitch_trait_test!(
            diminish_pitch,
            chordseq![[8], [], [10, 14]],
            &2,
            chordseq![[4], [], [5, 7]]
        );
        pitch_trait_test!(
            diminish_pitch,
            chordseq![[10], [], [12, 17]],
            &2.5,
            chordseq![[4], [], [4, 6]]
        );
        pitch_trait_test!(
            diminish_pitch,
            chordseq![[12.0], [], [16.5, 19.5]],
            &3,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        pitch_trait_test!(
            diminish_pitch,
            chordseq![[12.0], [], [16.5, 19.5]],
            &3.0,
            chordseq![[4.0], [], [5.5, 6.5]]
        );
        pitch_trait_test!(
            diminish_pitch,
            melody![[8], [], [10, 14]],
            &2,
            melody![[4], [], [5, 7]]
        );
        pitch_trait_test!(
            diminish_pitch,
            melody![[10], [], [12, 17]],
            &2.5,
            melody![[4], [], [4, 6]]
        );
        pitch_trait_test!(
            diminish_pitch,
            melody![[12.0], [], [16.5, 19.5]],
            &3,
            melody![[4.0], [], [5.5, 6.5]]
        );
        pitch_trait_test!(
            diminish_pitch,
            melody![[12.0], [], [16.5, 19.5]],
            &3.0,
            melody![[4.0], [], [5.5, 6.5]]
        );
    }

    #[test]
    fn trim_min() {
        pitch_trait_test!(trim_min, 5, 3, 5);
        pitch_trait_test!(trim_min, 3, 5, 5);
        pitch_trait_test!(trim_min, None, 5, None);
        pitch_trait_test!(trim_min, Some(5), 3, Some(5));
        pitch_trait_test!(trim_min, Some(3), 5, Some(5));
        pitch_trait_test!(trim_min, vec![5.0, 3.0], 4.0, vec![5.0, 4.0]);
        pitch_trait_test!(
            trim_min,
            MelodyMember::from(vec![5.0, 3.0]),
            4.0,
            MelodyMember::from(vec![5.0, 4.0])
        );
        pitch_trait_test!(trim_min, numseq![5.0, 3.0], 4.0, numseq![5.0, 4.0]);
        pitch_trait_test!(
            trim_min,
            noteseq![5.0, None, 3.0],
            4.0,
            noteseq![5.0, None, 4.0]
        );
        pitch_trait_test!(
            trim_min,
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[5.0], [], [4.5, 4.0]]
        );
        pitch_trait_test!(
            trim_min,
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[5.0], [], [4.5, 4.0]]
        );
    }

    #[test]
    fn trim_max() {
        pitch_trait_test!(trim_max, 5, 3, 3);
        pitch_trait_test!(trim_max, 3, 5, 3);
        pitch_trait_test!(trim_max, None, 5, None);
        pitch_trait_test!(trim_max, Some(5), 3, Some(3));
        pitch_trait_test!(trim_max, Some(3), 5, Some(3));
        pitch_trait_test!(trim_max, vec![5.0, 3.0], 4.0, vec![4.0, 3.0]);
        pitch_trait_test!(
            trim_max,
            MelodyMember::from(vec![5.0, 3.0]),
            4.0,
            MelodyMember::from(vec![4.0, 3.0])
        );
        pitch_trait_test!(trim_max, numseq![5.0, 3.0], 4.0, numseq![4.0, 3.0]);
        pitch_trait_test!(
            trim_max,
            noteseq![5.0, None, 3.0],
            4.0,
            noteseq![4.0, None, 3.0]
        );
        pitch_trait_test!(
            trim_max,
            chordseq![[5.0], [], [4.5, 3.0]],
            4.0,
            chordseq![[4.0], [], [4.0, 3.0]]
        );
        pitch_trait_test!(
            trim_max,
            melody![[5.0], [], [4.5, 3.0]],
            4.0,
            melody![[4.0], [], [4.0, 3.0]]
        );
    }

    #[test]
    fn trim() {
        macro_rules! trim_test {
            ($init:expr, $first:expr, $second:expr, $ret:expr) => {
                pitch_trait_test!(trim, $init, $first, $second, $ret);
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
