use crate::collections::traits::{Collection, CollectionError};
use crate::entities::scale::Scale;
use crate::ops::pitch::Pitch;
use anyhow::{anyhow, Result};
use num_traits::{Bounded, FromPrimitive, Num, PrimInt};
use std::fmt::Debug;
use std::iter::{zip, Sum};
use std::ops::SubAssign;
use std::slice::Iter;

#[macro_export]
macro_rules! sequence_pitch_methods {
    ($ty:ident) => {
        impl<T: Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum> Pitch<T>
            for $ty<T>
        {
            fn set_pitches(mut self, v: Vec<T>) -> Result<Self> {
                for p in self.contents.iter_mut() {
                    p.set_pitches(v.clone())?;
                }
                Ok(self)
            }

            fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.map_pitch(&f);
                });
                self
            }

            fn filter_pitch<FilterT: Fn(&T) -> bool>(mut self, f: FilterT) -> Result<Self> {
                for p in self.contents.iter_mut() {
                    p.filter_pitch(&f)?;
                }
                Ok(self)
            }

            fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
                for p in self.contents.iter_mut() {
                    p.filter_map_pitch(&f)?;
                }
                Ok(self)
            }

            fn transpose_pitch(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.transpose_pitch(v);
                });
                self
            }

            fn invert_pitch(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.invert_pitch(v);
                });
                self
            }

            fn modulus(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.modulus(v);
                });
                self
            }

            fn trim_min(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.trim_min(v);
                });
                self
            }

            fn trim_max(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.trim_max(v);
                });
                self
            }

            fn trim(mut self, first: T, second: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.trim(first, second);
                });
                self
            }

            fn bounce_min(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.bounce_min(v);
                });
                self
            }

            fn bounce_max(mut self, v: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.bounce_max(v);
                });
                self
            }

            fn bounce(mut self, first: T, second: T) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.bounce(first, second);
                });
                self
            }

            fn augment_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.augment_pitch(n);
                });
                self
            }

            fn diminish_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
                self.contents.iter_mut().for_each(|p| {
                    p.diminish_pitch(n);
                });
                self
            }

            fn is_silent(mut self) -> bool {
                self.contents.iter_mut().all(|m| m.is_silent())
            }
        }
    };
}

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
    T: Clone + Debug + PartialEq,
    PitchType: Pitch<PitchType> + Clone + Copy + Debug + FromPrimitive + Bounded + Num + PartialOrd + Sum,
>: Pitch<PitchType> + Collection<T>
{
    /// Modify all pitches in the Sequence by passing a mutable reference to them to the
    /// function passed in the argument.
    fn mutate_pitches<F: Fn(&mut PitchType)>(self, f: F) -> Self;

    /// Return a Vec of pitches in the order they appear in the Sequence.
    fn to_flat_pitches(&self) -> Vec<PitchType>;

    /// Return a Vec containing the pitches for each member of this Sequence.
    fn to_pitches(&self) -> Vec<Vec<PitchType>>;

    /// If all members of this sequence contain exactly one pitch, return a result containing a
    /// Vec of these pitches, otherwise return an Error.
    fn to_numeric_values(&self) -> Result<Vec<PitchType>>;

    /// If all members of this sequence contain zero or one pitches, return a result containing a
    /// Vec of these pitches, expressed as Options, otherwise return an Error.
    fn to_optional_numeric_values(&self) -> Result<Vec<Option<PitchType>>>;

    /// A map operation on pitches where the pitch is passed to the mapper function
    /// in a tuple of (position, pitch).
    fn map_pitch_enumerated<MapT: Fn((usize, &PitchType)) -> PitchType>(self, f: MapT) -> Self;

    /// A filter operation on pitches where the pitch is passed to the mapper function
    /// in a tuple of (position, pitch).
    fn filter_pitch_enumerated<FilterT: Fn((usize, &PitchType)) -> bool>(
        self,
        f: FilterT,
    ) -> Result<Self>;

    /// A filter_map operation on pitches where the pitch is passed to the mapper function
    /// in a tuple of (position, pitch).
    fn filter_map_pitch_enumerated<MapT: Fn((usize, &PitchType)) -> Option<PitchType>>(
        self,
        f: MapT,
    ) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Return an Option containing the lowest pitch in this Sequence, if any.
    fn min_value(&self) -> Option<PitchType> {
        self.to_flat_pitches()
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Return an Option containing the highest pitch in this Sequence, if any.
    fn max_value(&self) -> Option<PitchType> {
        self.to_flat_pitches()
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Return an Option containing the sum of all pitches in this Sequence, if any.
    fn total(&self) -> Option<PitchType> {
        Some(self.to_flat_pitches().iter().copied().sum())
    }

    /// Return an Option containing the mean value of all pitches in this Sequence, if any.
    fn mean(&self) -> Option<PitchType> {
        let pitches = self.to_flat_pitches();
        let mut iter = pitches.iter();
        let first = iter.next()?;

        Some(iter.fold(*first, |acc, x| acc + *x) / PitchType::from_usize(pitches.len())?)
    }

    /// Return an Option containing the distance between maximum and minimum pitches in
    /// this Sequence, if any.
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

    fn transpose_to_min(self, t: PitchType) -> Self {
        match self.min_value() {
            Some(m) => self.transpose_pitch(t - m),
            None => self,
        }
    }

    fn transpose_to_max(self, t: PitchType) -> Self {
        match self.max_value() {
            Some(m) => self.transpose_pitch(t - m),
            None => self,
        }
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
            c.splice(0..0, std::iter::repeat_n(val, num));
        })
    }

    fn pad_right(self, val: T, num: usize) -> Self {
        self.mutate_contents(|c| {
            c.extend(std::iter::repeat_n(val, num));
        })
    }

    fn pad_to(self, val: T, num: usize) -> Self {
        let len = self.length();

        if len < num {
            self.pad(val, num - len)
        } else {
            self
        }
    }

    fn pad_right_to(self, val: T, num: usize) -> Self {
        let len = self.length();

        if len < num {
            self.pad_right(val, num - len)
        } else {
            self
        }
    }

    fn dedupe(self) -> Self {
        self.mutate_contents(|c| {
            c.dedup();
        })
    }

    fn repeat(self) -> Self {
        self.mutate_contents(|c| c.extend(c.clone()))
    }

    fn repeated(self, num: usize) -> Self {
        self.replace_contents(|c| std::iter::repeat_n(c.clone(), num).flatten().collect())
    }

    fn looped(self, num: usize) -> Result<Self> {
        let len = self.length();

        if len == 0 {
            return Err(anyhow!(CollectionError::ZeroLength));
        }

        Ok(self.repeated(num / len + 1).mutate_contents(|c| {
            c.truncate(num);
        }))
    }

    fn looped_from(self, num: usize, start: i32) -> Result<Self> {
        let first = self.index(start)?;
        let last = num + first;
        let len = self.length();

        if len == 0 {
            return Err(anyhow!(CollectionError::ZeroLength));
        }

        Ok(self.repeated(last / len + 1).mutate_contents(|c| {
            c.truncate(last);
            c.drain(..first);
        }))
    }

    fn scale(self, scale: Scale<PitchType>, zeroval: PitchType) -> Result<Self>
    where
        PitchType: PrimInt
            + From<i8>
            + TryFrom<usize>
            + TryInto<usize>
            + Debug
            + Num
            + Sum
            + SubAssign
            + num_traits::Euclid,
    {
        Ok(self.mutate_pitches(scale.fit_to_scale(&zeroval)?))
    }

    fn combine(self, f: impl Fn((&T, &T)) -> T, seq: Self) -> Result<Self> {
        if self.length() != seq.length() {
            return Err(anyhow!(CollectionError::DifferentLengths(
                self.length(),
                seq.length()
            )));
        }

        let ret = zip(self.cts_ref(), seq.cts_ref()).map(f).collect();

        Ok(self.with_contents(ret))
    }

    fn map_with(self, f: impl Fn(Vec<&T>) -> T, seq: Vec<Self>) -> Result<Self> {
        let len = self.length();

        for s in seq.iter() {
            if len != s.length() {
                return Err(anyhow!(CollectionError::DifferentLengths(len, s.length())));
            }
        }

        let mut iters: Vec<Iter<'_, T>> = seq.iter().map(|v| v.cts_ref().iter()).collect();
        iters.insert(0, self.cts_ref().iter());

        let ret: Vec<T> = (0..len)
            .map(|_| f(iters.iter_mut().map(|n| n.next().unwrap()).collect()))
            .collect();

        Ok(self.with_contents(ret))
    }

    fn map_with_enumerated(
        self,
        f: impl Fn((usize, Vec<&T>)) -> T,
        seq: Vec<Self>,
    ) -> Result<Self> {
        let len = self.length();

        for s in seq.iter() {
            if len != s.length() {
                return Err(anyhow!(CollectionError::DifferentLengths(len, s.length())));
            }
        }

        let mut iters: Vec<Iter<'_, T>> = seq.iter().map(|v| v.cts_ref().iter()).collect();
        iters.insert(0, self.cts_ref().iter());

        let ret: Vec<T> = (0..len)
            .map(|i| f((i, iters.iter_mut().map(|n| n.next().unwrap()).collect())))
            .collect();

        Ok(self.with_contents(ret))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::assert_f64_near;

    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::{chordseq, melody, noteseq, numseq, scale};

    macro_rules! assert_numseq_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents.clone();
            let exp = $exp.contents.clone();

            assert!(val.len() == exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                assert_f64_near!(*a, *b, 40);
            }
        };
    }

    macro_rules! assert_noteseq_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents.clone();
            let exp = $exp.contents.clone();

            assert!(val.len() == exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                if a.is_none() {
                    assert!(
                        b.is_none(),
                        "result contained a value where it was not expected"
                    );
                } else {
                    assert!(
                        b.is_some(),
                        "result did not contain a value where it was expected"
                    );

                    assert_f64_near!(a.unwrap(), b.unwrap(), 40);
                }
            }
        };
    }

    macro_rules! assert_chordseq_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents.clone();
            let exp = $exp.contents.clone();

            assert_eq!(val.len(), exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                assert_eq!(a.len(), b.len(), "chords have different lengths");

                for (n1, n2) in a.iter().zip(b.iter()) {
                    assert_f64_near!(*n1, *n2, 40);
                }
            }
        };
    }

    macro_rules! assert_melody_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents.clone();
            let exp = $exp.contents.clone();

            assert_eq!(val.len(), exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                assert_eq!(
                    a.values.len(),
                    b.values.len(),
                    "chords have different lengths"
                );

                for (n1, n2) in a.values.iter().zip(b.values.iter()) {
                    assert_f64_near!(*n1, *n2, 40);
                }
            }
        };
    }

    #[test]
    fn set_pitches() {
        assert!(numseq![4, 2, 5, 6, 3].set_pitches(vec![]).is_err());
        assert!(numseq![4, 2, 5, 6, 3].set_pitches(vec![10, 12]).is_err());
        assert_eq!(
            numseq![4, 2, 5, 6, 3].set_pitches(vec![10]).unwrap(),
            numseq![10, 10, 10, 10, 10]
        );
        assert!(noteseq![4, 2, 5, 6, 3].set_pitches(vec![10, 12]).is_err());
        assert_eq!(
            noteseq![4, 2, 5, 6, 3].set_pitches(vec![]).unwrap(),
            noteseq![None, None, None, None, None]
        );
        assert_eq!(
            noteseq![4, 2, 5, 6, 3].set_pitches(vec![10]).unwrap(),
            noteseq![10, 10, 10, 10, 10]
        );
        assert_eq!(
            chordseq![4, 2, 5, 6, 3].set_pitches(vec![]).unwrap(),
            chordseq![[], [], [], [], []]
        );
        assert_eq!(
            chordseq![4, 2, 5, 6, 3].set_pitches(vec![10]).unwrap(),
            chordseq![[10], [10], [10], [10], [10]]
        );
        assert_eq!(
            chordseq![4, 2, 5, 6, 3].set_pitches(vec![10, 12]).unwrap(),
            chordseq![[10, 12], [10, 12], [10, 12], [10, 12], [10, 12]]
        );
    }

    #[test]
    fn map_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].map_pitch(|v| v + 1),
            numseq![5, 3, 6, 7, 4]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].map_pitch(|v| v + 1),
            noteseq![5, None, 6, 7, 4]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]].map_pitch(|v| v + 1),
            chordseq![[5], [], [6], [7, 4]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]].map_pitch(|v| v + 1),
            melody![[5], [], [6], [7, 4]]
        );
    }

    #[test]
    fn filter_pitch() {
        assert!(numseq![4, 2, 5, 6, 3].filter_pitch(|v| *v > 3).is_err());
        assert_eq!(
            numseq![4, 2, 5, 6, 3].filter_pitch(|v| *v > 1).unwrap(),
            numseq![4, 2, 5, 6, 3]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].filter_pitch(|v| *v > 3).unwrap(),
            noteseq![4, None, 5, 6, None]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]]
                .filter_pitch(|v| *v > 3)
                .unwrap(),
            chordseq![[4], [], [5], [6]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]]
                .filter_pitch(|v| *v > 3)
                .unwrap(),
            melody![[4], [], [5], [6]]
        );
    }

    #[test]
    fn filter_map_pitch() {
        fn fmfunc(v: &i32) -> Option<i32> {
            if *v > 3 {
                Some(*v + 1)
            } else {
                None
            }
        }
        assert!(numseq![4, 2, 5, 6, 3].filter_map_pitch(fmfunc).is_err());
        assert_eq!(
            numseq![4, 2, 5, 6, 3]
                .filter_map_pitch(|v| Some(*v + 1))
                .unwrap(),
            numseq![5, 3, 6, 7, 4]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].filter_map_pitch(fmfunc).unwrap(),
            noteseq![5, None, 6, 7, None]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]]
                .filter_map_pitch(fmfunc)
                .unwrap(),
            chordseq![[5], [], [6], [7]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]]
                .filter_map_pitch(fmfunc)
                .unwrap(),
            melody![[5], [], [6], [7]]
        );
    }

    #[test]
    fn transpose_pitch() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].transpose_pitch(5),
            numseq![9, 7, 10, 11, 8]
        );
        assert_noteseq_f64_near!(
            noteseq![4.0, None, 5.0, 6.0, 3.0].transpose_pitch(-5.5),
            noteseq![-1.5, None, -0.5, 0.5, -2.5]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]].transpose_pitch(5),
            chordseq![[9], [], [10], [11, 8]]
        );
        assert_melody_f64_near!(
            melody![[4.0], [], [5.0], [6.0, 3.0]].transpose_pitch(-5.5),
            melody![[-1.5], [], [-0.5], [0.5, -2.5]]
        );
    }

    #[test]
    fn invert_pitch() {
        assert_numseq_f64_near!(
            numseq![4.0, 2.0, 5.0, 6.0, 3.0].invert_pitch(5.5),
            numseq![7.0, 9.0, 6.0, 5.0, 8.0]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].invert_pitch(5),
            noteseq![6, None, 5, 4, 7]
        );
        assert_chordseq_f64_near!(
            chordseq![[4.0], [], [5.0], [6.0, 3.0]].invert_pitch(-5.5),
            chordseq![[-15.0], [], [-16.0], [-17.0, -14.0]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]].invert_pitch(5),
            melody![[6], [], [5], [4, 7]]
        );
    }

    #[test]
    fn modulus() {
        assert_numseq_f64_near!(
            numseq![4.0, 2.0, 5.0, -6.0, -3.0].modulus(2.5),
            numseq![1.5, 2.0, 0.0, 1.5, 2.0]
        );
        assert_eq!(
            noteseq![4, None, 5, -6, -3].modulus(4),
            noteseq![0, None, 1, 2, 1]
        );
        assert_chordseq_f64_near!(
            chordseq![[4.0], [], [5.0], [-6.0, -3.0]].modulus(2.5),
            chordseq![[1.5], [], [0.0], [1.5, 2.0]]
        );
        assert_eq!(
            melody![[4], [], [5], [-6, -3]].modulus(4),
            melody![[0], [], [1], [2, 1]]
        );
    }

    #[test]
    fn trim_min() {
        assert_numseq_f64_near!(
            numseq![4.0, 2.0, 5.0, 6.0, 3.0].trim_min(5.5),
            numseq![5.5, 5.5, 5.5, 6.0, 5.5]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].trim_min(5),
            noteseq![5, None, 5, 6, 5]
        );
        assert_chordseq_f64_near!(
            chordseq![[4.0], [], [5.0], [6.0, 3.0]].trim_min(5.5),
            chordseq![[5.5], [], [5.5], [6.0, 5.5]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]].trim_min(5),
            melody![[5], [], [5], [6, 5]]
        );
    }

    #[test]
    fn trim_max() {
        assert_numseq_f64_near!(
            numseq![4.0, 2.0, 5.0, 6.0, 3.0].trim_max(5.5),
            numseq![4.0, 2.0, 5.0, 5.5, 3.0]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].trim_max(5),
            noteseq![4, None, 5, 5, 3]
        );
        assert_chordseq_f64_near!(
            chordseq![[4.0], [], [5.0], [6.0, 3.0]].trim_max(5.5),
            chordseq![[4.0], [], [5.0], [5.5, 3.0]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]].trim_max(5),
            melody![[4], [], [5], [5, 3]]
        );
    }

    #[test]
    fn trim() {
        assert_numseq_f64_near!(
            numseq![4.0, 2.0, 5.0, 6.0, 3.0].trim(4.0, 5.5),
            numseq![4.0, 4.0, 5.0, 5.5, 4.0]
        );
        assert_eq!(
            noteseq![4, None, 5, 6, 3].trim(4, 5),
            noteseq![4, None, 5, 5, 4]
        );
        assert_chordseq_f64_near!(
            chordseq![[4.0], [], [5.0], [6.0, 3.0]].trim(4.0, 5.5),
            chordseq![[4.0], [], [5.0], [5.5, 4.0]]
        );
        assert_eq!(
            melody![[4], [], [5], [6, 3]].trim(4, 5),
            melody![[4], [], [5], [5, 4]]
        );
    }

    #[test]
    fn bounce_min() {
        assert_eq!(numseq![4, 2, 5, 6, 3].bounce_min(5), numseq![6, 8, 5, 6, 7]);
        assert_noteseq_f64_near!(
            noteseq![4.0, None, 5.0, 6.0, 3.0].bounce_min(5.5),
            noteseq![7.0, None, 6.0, 6.0, 8.0]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]].bounce_min(5),
            chordseq![[6], [], [5], [6, 7]]
        );
        assert_melody_f64_near!(
            melody![[4.0], [], [5.0], [6.0, 3.0]].bounce_min(5.5),
            melody![[7.0], [], [6.0], [6.0, 8.0]]
        );
    }

    #[test]
    fn bounce_max() {
        assert_eq!(numseq![4, 2, 5, 6, 3].bounce_max(5), numseq![4, 2, 5, 4, 3]);
        assert_noteseq_f64_near!(
            noteseq![4.0, None, 5.0, 6.0, 3.0].bounce_max(5.5),
            noteseq![4.0, None, 5.0, 5.0, 3.0]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]].bounce_max(5),
            chordseq![[4], [], [5], [4, 3]]
        );
        assert_melody_f64_near!(
            melody![[4.0], [], [5.0], [6.0, 3.0]].bounce_max(5.5),
            melody![[4.0], [], [5.0], [5.0, 3.0]]
        );
    }

    #[test]
    fn bounce() {
        assert_eq!(numseq![4, 2, 5, 6, 3].bounce(4, 5), numseq![4, 4, 5, 4, 5]);
        assert_noteseq_f64_near!(
            noteseq![4.0, None, 5.0, 6.0, 3.0].bounce(4.0, 5.5),
            noteseq![4.0, None, 5.0, 5.0, 5.0]
        );
        assert_eq!(
            chordseq![[4], [], [5], [6, 3]].bounce(4, 5),
            chordseq![[4], [], [5], [4, 5]]
        );
        assert_melody_f64_near!(
            melody![[4.0], [], [5.0], [6.0, 3.0]].bounce(4.0, 5.5),
            melody![[4.0], [], [5.0], [5.0, 5.0]]
        );
    }

    #[test]
    fn min_value() {
        assert!(NumericSeq::<i64>::new(vec![]).min_value().is_none());
        assert_eq!(numseq![4, 2, 5, 6, 3].min_value(), Some(2));
        assert_eq!(noteseq![4.1, 2.8, 5.4, 6.3, 3.0].min_value(), Some(2.8));
    }

    #[test]
    fn max_value() {
        assert!(NumericSeq::<i64>::new(vec![]).max_value().is_none());
        assert_eq!(numseq![4, 2, 5, 6, 3].max_value(), Some(6));
        assert_eq!(chordseq![4.1, 2.8, 5.4, 6.3, 3.0].max_value(), Some(6.3));
    }

    #[test]
    fn to_optional_numeric_values() {
        assert_eq!(
            numseq![4, 2, 5, 6, 3].to_optional_numeric_values().unwrap(),
            vec![Some(4), Some(2), Some(5), Some(6), Some(3)]
        );
    }

    #[test]
    fn find_if_window() {
        assert_eq!(
            numseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_window(2, 1, |s| s[0] == s[1]),
            vec![0, 3, 4, 6, 8]
        );
        assert_eq!(
            noteseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_window(2, 2, |s| s[0] == s[1]),
            vec![0, 4, 6, 8]
        );
        assert_eq!(
            numseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_window(1, 2, |s| s[0] % 2 == 0),
            vec![2, 6]
        );
    }

    #[test]
    fn find_if_reverse_window() {
        assert_eq!(
            numseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_reverse_window(2, 1, |s| s[0] == s[1]),
            vec![8, 6, 4, 3, 0]
        );
        assert_eq!(
            noteseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_reverse_window(2, 2, |s| s[0] == s[1]),
            vec![8, 6, 4, 0]
        );
        assert_eq!(
            numseq![1, 1, 2, 3, 3, 3, 4, 4, 5, 5].find_if_reverse_window(1, 2, |s| s[0] % 2 == 0),
            vec![7]
        );
    }

    #[test]
    fn transpose_to_min() {
        assert_eq!(
            NumericSeq::<i32>::new(vec![]).transpose_to_min(44),
            NumericSeq::<i32>::new(vec![])
        );
        assert_eq!(chordseq![1, 6, 4].transpose_to_min(2), chordseq![2, 7, 5]);

        assert_numseq_f64_near!(
            numseq![1.7, 3.4, 6.3].transpose_to_min(-1.8),
            numseq![-1.8, -0.1, 2.8]
        );
    }

    #[test]
    fn transpose_to_max() {
        assert_eq!(
            NumericSeq::new(vec![]).transpose_to_max(44),
            NumericSeq::new(vec![])
        );
        assert_eq!(chordseq![1, 6, 4].transpose_to_max(2), chordseq![-3, 2, 0]);

        assert_numseq_f64_near!(
            numseq![1.7, 3.4, 6.3].transpose_to_max(-1.8),
            numseq![-6.4, -4.7, -1.8]
        );
    }

    #[test]
    fn scale() {
        let chromatic = scale!["chromatic"].unwrap();
        let lydian = scale!["lydian"].unwrap();

        let v64: Vec<i64> = (-20..20).collect();

        assert_eq!(
            NumericSeq::new(v64.clone()).scale(chromatic, 60).unwrap(),
            NumericSeq::new((40..80).collect::<Vec<i64>>())
        );

        assert_eq!(
            NumericSeq::new(v64).scale(lydian.clone(), 60).unwrap(),
            numseq![
                26, 28, 30, 31, 33, 35, 36, 38, 40, 42, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60,
                62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91, 93
            ]
        );

        assert_eq!(
            noteseq![-1, None, 0, 1].scale(lydian, 20).unwrap(),
            noteseq![19, None, 20, 22]
        );
    }

    #[test]
    fn flat_map_windows() {
        assert_eq!(
            numseq![1, 2, 3, 4, 5]
                .flat_map_windows(2, 1, |mut w| {
                    w.reverse();
                    w
                })
                .unwrap(),
            numseq![2, 1, 3, 2, 4, 3, 5, 4]
        );

        assert_eq!(
            noteseq![1, None, 3, 4, 5]
                .flat_map_windows(3, 2, |mut w| {
                    w.reverse();
                    w
                })
                .unwrap(),
            noteseq![3, None, 1, 5, 4, 3]
        );
    }

    #[test]
    fn filter_windows() {
        assert_eq!(
            numseq![1, 2, 3, 4, 5]
                .filter_windows(2, 1, |w| w[0] > 1)
                .unwrap(),
            numseq![2, 3, 3, 4, 4, 5]
        );

        assert_eq!(
            numseq![1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
                .filter_windows(3, 2, |w| w[0] > 2)
                .unwrap(),
            numseq![3, 4, 5, 5, 6, 5, 5, 4, 3, 3, 2, 1]
        );
    }

    #[test]
    fn pad() {
        assert_eq!(numseq![1, 2, 3].pad(4, 1), numseq![4, 1, 2, 3]);

        assert_eq!(
            noteseq![1, None, 3].pad(Some(2), 4),
            noteseq![2, 2, 2, 2, 1, None, 3]
        );
    }

    #[test]
    fn pad_right() {
        assert_eq!(numseq![1, 2, 3].pad_right(4, 1), numseq![1, 2, 3, 4]);

        assert_eq!(
            noteseq![1, None, 3].pad_right(Some(2), 4),
            noteseq![1, None, 3, 2, 2, 2, 2]
        );
    }

    #[test]
    fn pad_to() {
        assert_eq!(numseq![1, 2, 3].pad_to(4, 3), numseq![1, 2, 3]);

        assert_eq!(noteseq![1, 2, 3].pad_to(Some(4), 4), noteseq![4, 1, 2, 3]);

        assert_eq!(numseq![1, 2, 3].pad_to(2, 5), numseq![2, 2, 1, 2, 3]);
    }

    #[test]
    fn pad_right_to() {
        assert_eq!(numseq![1, 2, 3].pad_right_to(4, 3), numseq![1, 2, 3]);

        assert_eq!(
            noteseq![1, 2, 3].pad_right_to(Some(4), 4),
            noteseq![1, 2, 3, 4]
        );

        assert_eq!(numseq![1, 2, 3].pad_right_to(2, 5), numseq![1, 2, 3, 2, 2]);
    }

    #[test]
    fn dedupe() {
        assert_eq!(
            numseq![1, 2, 2, 3, 1, 3, 3, 2].dedupe(),
            numseq![1, 2, 3, 1, 3, 2]
        );
    }

    #[test]
    fn repeat() {
        assert_eq!(numseq![1, 2, 3].repeat(), numseq![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn repeated() {
        assert_eq!(numseq![1, 2, 3].repeated(0), numseq![]);
        assert_eq!(
            noteseq![1, None, 3].repeated(2),
            noteseq![1, None, 3, 1, None, 3]
        );
    }

    #[test]
    fn looped() {
        assert!(NumericSeq::<i32>::new(vec![]).looped(3).is_err());

        assert_eq!(numseq![1, 2, 3].looped(0).unwrap(), numseq![]);
        assert_eq!(numseq![1, 2, 3].looped(2).unwrap(), numseq![1, 2]);
        assert_eq!(numseq![1, 2, 3].looped(3).unwrap(), numseq![1, 2, 3]);
        assert_eq!(numseq![1, 2, 3].looped(4).unwrap(), numseq![1, 2, 3, 1]);
        assert_eq!(numseq![1, 2, 3].looped(5).unwrap(), numseq![1, 2, 3, 1, 2]);
        assert_eq!(
            numseq![1, 2, 3].looped(6).unwrap(),
            numseq![1, 2, 3, 1, 2, 3]
        );
        assert_eq!(
            numseq![1, 2, 3].looped(7).unwrap(),
            numseq![1, 2, 3, 1, 2, 3, 1]
        );
    }

    #[test]
    fn looped_from() {
        assert!(NumericSeq::<i32>::new(vec![]).looped_from(3, 0).is_err());
        assert!(numseq![1, 2, 3].looped_from(3, 3).is_err());
        assert!(numseq![1, 2, 3].looped_from(3, -4).is_err());

        assert_eq!(numseq![1, 2, 3].looped_from(0, -3).unwrap(), numseq![]);
        assert_eq!(numseq![1, 2, 3].looped_from(2, -1).unwrap(), numseq![3, 1]);
        assert_eq!(
            numseq![1, 2, 3].looped_from(3, 0).unwrap(),
            numseq![1, 2, 3]
        );
        assert_eq!(
            numseq![1, 2, 3].looped_from(4, 2).unwrap(),
            numseq![3, 1, 2, 3]
        );
        assert_eq!(
            numseq![1, 2, 3].looped_from(5, 0).unwrap(),
            numseq![1, 2, 3, 1, 2]
        );
        assert_eq!(
            numseq![1, 2, 3].looped_from(6, -1).unwrap(),
            numseq![3, 1, 2, 3, 1, 2]
        );
        assert_eq!(
            numseq![1, 2, 3].looped_from(7, 0).unwrap(),
            numseq![1, 2, 3, 1, 2, 3, 1]
        );
    }

    #[test]
    fn test_combine() {
        assert!(numseq![1, 2, 3]
            .combine(|(a, b)| a + b, numseq![4, 5])
            .is_err());

        assert_eq!(
            numseq![1, 2, 3]
                .combine(|(a, b)| a + b, numseq![4, 5, 6])
                .unwrap(),
            numseq![5, 7, 9]
        );
    }

    #[test]
    fn test_map_with() {
        assert!(numseq![1, 2, 3]
            .map_with(
                |v| v.into_iter().sum(),
                vec![numseq![4, 5], numseq![7, 8, 9]]
            )
            .is_err());

        assert_eq!(
            numseq![1, 2, 3]
                .map_with(
                    |v| v.into_iter().sum(),
                    vec![numseq![4, 5, 6], numseq![7, 8, 9]]
                )
                .unwrap(),
            numseq![12, 15, 18]
        );
    }

    #[test]
    fn test_map_with_enumerated() {
        assert!(numseq![1, 2, 3]
            .map_with_enumerated(
                |(i, v)| v.into_iter().sum::<i32>() * i as i32,
                vec![numseq![4, 5], numseq![7, 8, 9]]
            )
            .is_err());

        assert_eq!(
            numseq![1, 2, 3]
                .map_with_enumerated(
                    |(i, v)| v.into_iter().sum::<i32>() * i as i32,
                    vec![numseq![4, 5, 6], numseq![7, 8, 9]]
                )
                .unwrap(),
            numseq![0, 15, 36]
        );
    }
}
