use crate::sequences::{
    chord::ChordSeq,
    melody::{Melody, MelodyMember},
    note::NoteSeq,
    numeric::NumericSeq,
};

pub trait Pitch<T> {
    fn transpose_pitch(&mut self, n: T);
    fn invert_pitch(&mut self, n: T);
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

macro_rules! impl_traits_for_raw_values {
    (for $($ty:ident)*) => ($(
        impl Pitch<$ty> for $ty {
            fn transpose_pitch(&mut self, n: $ty) {
                *self += n
            }

            fn invert_pitch(&mut self, n: $ty) {
                *self = n + n - *self
            }

            fn is_silent(&self) -> bool {
                false
            }
        }

        impl Pitch<$ty> for Option<$ty> {
            impl_fns_for_option!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.is_none()
            }
        }

        impl Pitch<$ty> for Vec<$ty> {
            impl_fns_for_vec!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.is_empty()
            }
        }

        impl Pitch<$ty> for MelodyMember<$ty> {
            impl_fns_for_melody_member!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.values.is_empty() || self.volume == 0
            }
        }

        impl Pitch<$ty> for NumericSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.contents.is_empty()
            }
        }

        impl Pitch<$ty> for NoteSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }

        impl Pitch<$ty> for ChordSeq<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }

        impl Pitch<$ty> for Melody<$ty> {
            impl_fns_for_seq!($ty, for transpose_pitch invert_pitch);

            fn is_silent(&self) -> bool {
                self.contents.iter().all(|m| m.is_silent())
            }
        }
    )*)
}

impl_traits_for_raw_values!(for i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

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
