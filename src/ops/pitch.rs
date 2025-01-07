use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
    traits::Sequence,
};

pub trait Pitch<T> {
    fn transpose_pitch(&mut self, n: T);
    fn invert_pitch(&mut self, n: T);
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

macro_rules! impl_traits_for_raw_values {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn transpose_pitch(&mut self, n: $ty) {
                *self += n
            }

            fn invert_pitch(&mut self, n: $ty) {
                *self = n + n - *self
            }
        }

        impl Pitch<$ty> for Option<$ty> {
            impl_fns_for_option!($ty, for transpose_pitch invert_pitch);
        }

        impl Pitch<$ty> for Vec<$ty> {
            impl_fns_for_vec!($ty, for transpose_pitch invert_pitch);
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            impl_fns_for_melody_member!($ty, for transpose_pitch invert_pitch);
        }

        impl Pitch<$ty> for NumericSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);
        }

        impl Pitch<$ty> for NoteSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);
        }

        impl Pitch<$ty> for ChordSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);
        }
    )*)
}

impl_traits_for_raw_values!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::traits::Collection;

    macro_rules! pitch_trait_test {
        ($fn:ident, $init:expr, $arg:expr, $ret:expr) => {
            let mut x = $init;
            x.$fn($arg);
            assert_eq!(x, $ret);
        };
    }

    macro_rules! transpose_test {
        ($init:expr, $arg:expr, $ret:expr) => {
            let mut x = $init;
            x.transpose_pitch($arg);
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
        pitch_trait_test!(
            transpose_pitch,
            NumericSeq::new(vec![4, 5, 6]),
            6,
            NumericSeq::new(vec![10, 11, 12])
        );
        pitch_trait_test!(
            transpose_pitch,
            NoteSeq::new(vec![Some(4), None, Some(6)]),
            6,
            NoteSeq::new(vec![Some(10), None, Some(12)])
        );
        pitch_trait_test!(
            transpose_pitch,
            ChordSeq::new(vec![vec![4, 5, 6], vec![], vec![7]]),
            6,
            ChordSeq::new(vec![vec![10, 11, 12], vec![], vec![13]])
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
        pitch_trait_test!(
            invert_pitch,
            NumericSeq::new(vec![4, 5, 6]),
            6,
            NumericSeq::new(vec![8, 7, 6])
        );
        pitch_trait_test!(
            invert_pitch,
            NoteSeq::new(vec![Some(4), None, Some(6)]),
            6,
            NoteSeq::new(vec![Some(8), None, Some(6)])
        );
        pitch_trait_test!(
            invert_pitch,
            ChordSeq::new(vec![vec![4, 5, 6], vec![], vec![7]]),
            6,
            ChordSeq::new(vec![vec![8, 7, 6], vec![], vec![5]])
        );
    }
}
