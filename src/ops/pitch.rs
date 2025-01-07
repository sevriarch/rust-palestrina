use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
    traits::Sequence,
};

use std::ops::{Add, Sub};

trait Pitch<T> {
    fn invert(&mut self, n: T);
}

macro_rules! impl_traits_for_raw_values {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn invert(&mut self, n: $ty) {
                *self = n + n -*self
            }
        }

        impl Pitch<$ty> for Option<$ty> {
            fn invert(&mut self, n: $ty) {
                if let Some(p) = self.as_mut() {
                    p.invert(n);
                }
            }
        }

        impl Pitch<$ty> for Vec<$ty> {
            fn invert(&mut self, n: $ty) {
                self.iter_mut().for_each(|p| p.invert(n));
            }
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            fn invert(&mut self, n: $ty) {
                self.values.invert(n);
            }
        }
    )*)
}

impl_traits_for_raw_values!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert() {
        let mut x = 5;
        x.invert(6);
        assert_eq!(x, 7);

        let mut x = 5.5;
        x.invert(6.0);
        assert_eq!(x, 6.5);

        let mut x = Some(5);
        x.invert(6);
        assert_eq!(x, Some(7));

        let mut x = vec![4, 5, 6];
        x.invert(6);
        assert_eq!(x, vec![8, 7, 6]);

        let mut x = MelodyMember::from(vec![4, 5, 6]);
        x.invert(6);
        assert_eq!(x, MelodyMember::from(vec![8, 7, 6]));
    }
}
