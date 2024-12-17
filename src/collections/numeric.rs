use crate::collections::traits::Collection;

use num_traits::{Bounded, Num};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::Add;

#[derive(Clone, Debug, PartialEq)]
pub struct NumericSeq<T> {
    contents: Vec<T>,
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded> Collection<T> for NumericSeq<T> {
    fn new(contents: Vec<T>) -> NumericSeq<T> {
        Self { contents }
    }

    fn cts(&self) -> Vec<T> {
        self.contents.clone()
    }

    fn length(&self) -> usize {
        self.contents.len()
    }

    fn construct(&self, cts: Vec<T>) -> Box<NumericSeq<T>> {
        Box::new(NumericSeq::new(cts))
    }
}

impl<T> NumericSeq<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Add<Output = T> + Sum + Bounded + From<i32>,
{
    pub fn to_pitches(&self) -> Result<Vec<Vec<T>>, &str> {
        Ok(self.contents.clone().into_iter().map(|p| vec![p]).collect())
    }

    pub fn to_flat_pitches(&self) -> Result<Vec<T>, &str> {
        Ok(self.contents.clone())
    }

    pub fn to_numeric_values(&self) -> Result<Vec<T>, &str> {
        Ok(self.contents.clone())
    }

    pub fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, &str> {
        Ok(self.contents.clone().into_iter().map(|p| Some(p)).collect())
    }

    pub fn min(&self) -> Option<T> {
        self.contents.iter().copied().reduce(|a, b| if a < b { a } else { b })
    }

    pub fn max(&self) -> Option<T> {
        self.contents.iter().copied().reduce(|a, b| if a > b { a } else { b })
    }

    pub fn range(&self) -> Option<T> {
        if let (Some(min), Some(max)) = (self.min(), self.max()) {
            Some(max - min)
        } else {
            None
        }
    }

    pub fn total(&self) -> Option<T> {
        Some(self.contents.iter().copied().sum())
    }

    pub fn mean(&self) -> Option<T> {
        let mut iter = self.contents.iter();
        let first = iter.next()?;

        Some(iter.fold(*first, |acc, x| acc + *x) / T::from(self.contents.len() as i32))
    }

    pub fn find_if_window(&self, len: usize, step: usize, f: fn(&[T]) -> bool) -> Vec<usize> {
        let cts = self.cts();

        (0..=self.length() - len)
            .step_by(step)
            .filter(|i| f(&cts[*i..*i + len]))
            .collect()
    }

    pub fn find_if_reverse_window(
        &self,
        len: usize,
        step: usize,
        f: fn(&[T]) -> bool,
    ) -> Vec<usize> {
        let cts = self.cts();
        let maxposs = self.length() - len;

        (0..=self.length() - len)
            .step_by(step)
            .map(|i| maxposs - i)
            .filter(|i| f(&cts[*i..*i + len]))
            .collect()
    }

    pub fn map_pitches(mut self, f: impl Fn(T) -> T) -> Self {
        for v in self.contents.iter_mut() {
            *v = f(*v);
        }

        self
    }

    pub fn transpose(self, t: T) -> Self {
        self.map_pitches(|v| v + t)
    }

    pub fn transpose_to_min(self, t: T) -> Self {
        match self.min() {
            Some(m) => self.transpose(t - m),
            _ => self,
        }
    }

    pub fn transpose_to_max(self, t: T) -> Self {
        match self.max() {
            Some(m) => self.transpose(t - m),
            _ => self,
        }
    }

    pub fn invert(self, t: T) -> Self {
        self.map_pitches(|v| t + t - v)
    }

    pub fn augment(self, t: T) -> Self {
        self.map_pitches(|v | t * v)
    }

    // TODO: Handle division by zero
    pub fn diminish(self, t: T) -> Self {
        self.map_pitches(|v| v / t)
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::numeric::NumericSeq;
    use crate::collections::traits::Collection;

    use assert_float_eq::assert_f64_near;

    #[test]
    fn to_pitches() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_pitches(),
            Ok(vec![vec![4], vec![2], vec![5], vec![6], vec![3]])
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_flat_pitches(),
            Ok(vec![4, 2, 5, 6, 3])
        );
    }

    #[test]
    fn to_numeric_values() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_numeric_values(),
            Ok(vec![4, 2, 5, 6, 3])
        );
    }

    #[test]
    fn min() {
        assert!(NumericSeq::<i64>::new(vec![]).min().is_none());
        assert_eq!(NumericSeq::new(vec![4,2,5,6,3]).min(), Some(2));
        assert_eq!(NumericSeq::new(vec![4.1,2.8,5.4,6.3,3.0]).min(), Some(2.8));
    }

    #[test]
    fn max() {
        assert!(NumericSeq::<i64>::new(vec![]).max().is_none());
        assert_eq!(NumericSeq::new(vec![4,2,5,6,3]).max(), Some(6));
        assert_eq!(NumericSeq::new(vec![4.1,2.8,5.4,6.3,3.0]).max(), Some(6.3));
    }

    #[test]
    fn to_optional_numeric_values() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_optional_numeric_values(),
            Ok(vec![Some(4), Some(2), Some(5), Some(6), Some(3)])
        );
    }

    #[test]
    fn find_if_window() {
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_window(2, 1, |s| s[0] == s[1]),
            vec![0, 3, 4, 6, 8]
        );
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_window(2, 2, |s| s[0] == s[1]),
            vec![0, 4, 6, 8]
        );
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_window(1, 2, |s| s[0] % 2 == 0),
            vec![2, 6]
        );
    }

    #[test]
    fn find_if_reverse_window() {
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_reverse_window(2, 1, |s| s[0] == s[1]),
            vec![8, 6, 4, 3, 0]
        );
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_reverse_window(2, 2, |s| s[0] == s[1]),
            vec![8, 6, 4, 0]
        );
        assert_eq!(
            NumericSeq::new(vec![1, 1, 2, 3, 3, 3, 4, 4, 5, 5])
                .find_if_reverse_window(1, 2, |s| s[0] % 2 == 0),
            vec![7]
        );
    }

    // TODO: This macro needs a bit more introspection into results
    macro_rules! assert_contents_f64_near {
        ($val: expr, $exp: expr) => {
            let val = $val.contents;
            let exp = $exp.contents;

            assert!(val.len() == exp.len(), "lengths are different");
            for (a, b) in val.iter().zip(exp.iter()) {
                assert_f64_near!(*a, *b, 40);
            }
        }
    }

    #[test]
    fn transpose() {
        assert_eq!(NumericSeq::new(vec![1,6,4]).transpose(2), NumericSeq::new(vec![3,8,6]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .transpose(-1.8),
            NumericSeq::new(vec![-0.1,1.6,4.5]));
    }

    #[test]
    fn transpose_to_min() {
        assert_eq!(NumericSeq::new(vec![]).transpose_to_min(44), NumericSeq::new(vec![]));
        assert_eq!(NumericSeq::new(vec![1,6,4]).transpose_to_min(2), NumericSeq::new(vec![2,7,5]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .transpose_to_min(-1.8),
            NumericSeq::new(vec![-1.8,-0.1,2.8]));
    }

    #[test]
    fn transpose_to_max() {
        assert_eq!(NumericSeq::new(vec![]).transpose_to_max(44), NumericSeq::new(vec![]));
        assert_eq!(NumericSeq::new(vec![1,6,4]).transpose_to_max(2), NumericSeq::new(vec![-3,2,0]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .transpose_to_max(-1.8),
            NumericSeq::new(vec![-6.4,-4.7,-1.8]));
    }

    #[test]
    fn invert() {
        assert_eq!(NumericSeq::new(vec![1,6,4]).invert(2), NumericSeq::new(vec![3,-2,0]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .invert(-1.8),
            NumericSeq::new(vec![-5.3,-7.0,-9.9]));
    }

    #[test]
    fn augment() {
        assert_eq!(NumericSeq::new(vec![1,6,4]).augment(2), NumericSeq::new(vec![2,12,8]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .augment(2.0),
            NumericSeq::new(vec![3.4,6.8,12.6]));
    }

    #[test]
    fn diminish() {
        assert_eq!(NumericSeq::new(vec![1,6,4]).diminish(2), NumericSeq::new(vec![0,3,2]));

        assert_contents_f64_near!(NumericSeq::new(vec![1.7,3.4,6.3])
            .diminish(2.0),
            NumericSeq::new(vec![0.85,1.7,3.15]));
    }
}