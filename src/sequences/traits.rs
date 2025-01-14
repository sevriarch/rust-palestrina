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
    T: Clone + Debug + PartialEq,
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

    fn pad_right(self, val: T, num: usize) -> Self {
        self.mutate_contents(|c| {
            c.extend(std::iter::repeat(val).take(num));
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
        self.replace_contents(|c| std::iter::repeat(c.clone()).take(num).flatten().collect())
    }

    fn looped(self, num: usize) -> Result<Self> {
        let len = self.length();

        if len == 0 {
            return Err(anyhow!("cannot loop a zero-length sequence"));
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
            return Err(anyhow!("cannot loop a zero-length sequence"));
        }

        Ok(self.repeated(last / len + 1).mutate_contents(|c| {
            c.truncate(last);
            c.drain(..first);
        }))
    }

    fn scale(self, scale: Scale<PitchType>, zeroval: PitchType) -> Result<Self>
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::assert_f64_near;

    use crate::sequences::chord::ChordSeq;
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::{chordseq, noteseq, numseq};

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

    // TODO: This macro needs a bit more introspection into results
    macro_rules! assert_contents_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents.clone();
            let exp = $exp.contents.clone();

            assert!(val.len() == exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                assert_f64_near!(*a, *b, 40);
            }
        };
    }

    #[test]
    fn transpose() {
        assert_eq!(numseq![1, 6, 4].transpose(2).unwrap(), numseq![3, 8, 6]);

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].transpose(-1.8).unwrap(),
            numseq![-0.1, 1.6, 4.5]
        );
    }

    #[test]
    fn transpose_to_min() {
        assert_eq!(
            NumericSeq::<i32>::new(vec![]).transpose_to_min(44).unwrap(),
            NumericSeq::<i32>::new(vec![])
        );
        assert_eq!(
            chordseq![1, 6, 4].transpose_to_min(2).unwrap(),
            chordseq![2, 7, 5]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].transpose_to_min(-1.8).unwrap(),
            numseq![-1.8, -0.1, 2.8]
        );
    }

    #[test]
    fn transpose_to_max() {
        assert_eq!(
            NumericSeq::new(vec![]).transpose_to_max(44).unwrap(),
            NumericSeq::new(vec![])
        );
        assert_eq!(
            chordseq![1, 6, 4].transpose_to_max(2).unwrap(),
            chordseq![-3, 2, 0]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].transpose_to_max(-1.8).unwrap(),
            numseq![-6.4, -4.7, -1.8]
        );
    }

    #[test]
    fn invert() {
        assert_eq!(chordseq![1, 6, 4].invert(2).unwrap(), chordseq![3, -2, 0]);
        assert_eq!(
            noteseq![1, None, 6, 4].invert(2).unwrap(),
            noteseq![3, None, -2, 0]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].invert(-1.8).unwrap(),
            numseq![-5.3, -7.0, -9.9]
        );
    }

    #[test]
    fn augment() {
        assert_eq!(chordseq![1, 6, 4].augment(2).unwrap(), chordseq![2, 12, 8]);
        assert_eq!(
            noteseq![1, None, 6, 4].augment(2).unwrap(),
            noteseq![2, None, 12, 8]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].augment(2.0).unwrap(),
            numseq![3.4, 6.8, 12.6]
        );
    }

    #[test]
    fn diminish() {
        assert!(numseq![1, 6, 4].diminish(0).is_err());

        assert_eq!(chordseq![1, 6, 4].diminish(2).unwrap(), chordseq![0, 3, 2]);
        assert_eq!(
            noteseq![1, None, 6, 4].diminish(2).unwrap(),
            noteseq![0, None, 3, 2]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].diminish(2.0).unwrap(),
            numseq![0.85, 1.7, 3.15]
        );
    }

    #[test]
    fn modulus() {
        assert!(numseq![-1, 6, 4].modulus(0).is_err());
        assert_eq!(chordseq![-1, 6, 4].modulus(3).unwrap(), chordseq![2, 0, 1]);
        assert_eq!(
            noteseq![-1, None, 6, 4].modulus(3).unwrap(),
            noteseq![2, None, 0, 1]
        );

        assert_contents_f64_near!(
            numseq![-1.7, 3.4, 6.3].modulus(2.0).unwrap(),
            numseq![0.3, 1.4, 0.3]
        );
    }

    #[test]
    fn trim() {
        assert_eq!(
            noteseq![1, None, 2, 5, 6, 4].trim(2, 5).unwrap(),
            noteseq![2, None, 2, 5, 5, 4]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].trim(2.0, 5.0).unwrap(),
            numseq![2.0, 3.4, 5.0]
        );
    }

    #[test]
    fn trim_min() {
        assert_eq!(
            noteseq![1, None, 2, 5, 6, 4].trim_min(2).unwrap(),
            noteseq![2, None, 2, 5, 6, 4]
        );

        assert_contents_f64_near!(
            numseq![2.0, 3.4, 6.3].trim_min(2.0).unwrap(),
            numseq![2.0, 3.4, 6.3]
        );
    }

    #[test]
    fn trim_max() {
        assert_eq!(
            noteseq![1, None, 2, 5, 6, 4].trim_max(5).unwrap(),
            noteseq![1, None, 2, 5, 5, 4]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].trim_max(5.0).unwrap(),
            numseq![1.7, 3.4, 5.0]
        );
    }

    #[test]
    fn bounce() {
        assert_eq!(
            noteseq![0, None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                .bounce(7, 10)
                .unwrap(),
            noteseq![8, None, 7, 8, 9, 10, 9, 8, 7, 8, 9, 10, 9, 8, 7, 8, 9, 10, 9]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].bounce(2.0, 3.0).unwrap(),
            numseq![2.3, 2.6, 2.3]
        );
    }

    #[test]
    fn bounce_min() {
        assert_eq!(
            noteseq![1, None, 2, 5, 6, 4].bounce_min(2).unwrap(),
            noteseq![3, None, 2, 5, 6, 4]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].bounce_min(2.0).unwrap(),
            numseq![2.3, 3.4, 6.3]
        );
    }

    #[test]
    fn bounce_max() {
        assert_eq!(
            noteseq![1, None, 2, 5, 6, 4].bounce_max(5).unwrap(),
            noteseq![1, None, 2, 5, 4, 4]
        );

        assert_contents_f64_near!(
            numseq![1.7, 3.4, 6.3].bounce_max(5.0).unwrap(),
            numseq![1.7, 3.4, 3.7]
        );
    }

    #[test]
    fn scale() {
        let chromatic = Scale::new().with_name("chromatic").unwrap();
        let lydian = Scale::new().with_name("lydian").unwrap();

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
    fn test_map_with_indexed() {
        assert!(numseq![1, 2, 3]
            .map_with_indexed(
                |(i, v)| v.into_iter().sum::<i32>() * i as i32,
                vec![numseq![4, 5], numseq![7, 8, 9]]
            )
            .is_err());

        assert_eq!(
            numseq![1, 2, 3]
                .map_with_indexed(
                    |(i, v)| v.into_iter().sum::<i32>() * i as i32,
                    vec![numseq![4, 5, 6], numseq![7, 8, 9]]
                )
                .unwrap(),
            numseq![0, 15, 36]
        );
    }

    #[test]
    fn test_chaining() {
        fn chained_methods() -> Result<NumericSeq<i32>> {
            let ret = numseq![1, 2, 3].augment(3)?.transpose(1)?.invert(13)?;

            Ok(ret)
        }

        assert_eq!(chained_methods().unwrap(), numseq![22, 19, 16])
    }
}
