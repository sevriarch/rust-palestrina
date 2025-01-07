use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
    traits::Sequence,
};

trait Pitch<T> {
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
    )*)
}

impl_traits_for_raw_values!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_pitch() {
        let mut x = 5;
        x.transpose_pitch(6);
        assert_eq!(x, 11);

        let mut x = 5.5;
        x.transpose_pitch(6.0);
        assert_eq!(x, 11.5);

        let mut x: Option<i32> = None;
        x.transpose_pitch(6);
        assert_eq!(x, None);

        let mut x = Some(5);
        x.transpose_pitch(6);
        assert_eq!(x, Some(11));

        let mut x = vec![4, 5, 6];
        x.transpose_pitch(6);
        assert_eq!(x, vec![10, 11, 12]);

        let mut x = MelodyMember::from(vec![4, 5, 6]);
        x.transpose_pitch(6);
        assert_eq!(x, MelodyMember::from(vec![10, 11, 12]));
    }

    #[test]
    fn invert_pitch() {
        let mut x = 5;
        x.invert_pitch(6);
        assert_eq!(x, 7);

        let mut x = 5.5;
        x.invert_pitch(6.0);
        assert_eq!(x, 6.5);

        let mut x = Some(5);
        x.invert_pitch(6);
        assert_eq!(x, Some(7));

        let mut x = vec![4, 5, 6];
        x.invert_pitch(6);
        assert_eq!(x, vec![8, 7, 6]);

        let mut x = MelodyMember::from(vec![4, 5, 6]);
        x.invert_pitch(6);
        assert_eq!(x, MelodyMember::from(vec![8, 7, 6]));
    }
}
