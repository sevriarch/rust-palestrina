use crate::collections::traits::Collection;

use num_traits::{Bounded, Num};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::Add;

pub struct NumericSeq<T> {
    contents: Vec<T>,
}

impl<T: Clone + Copy + Num + Debug + Ord + Bounded> Collection<T> for NumericSeq<T> {
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
    T: Clone + Copy + Num + Debug + Ord + Add<Output = T> + Sum + Bounded + From<i32>,
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

    pub fn to_optional_numeric_values(&self) -> Result<Option<Vec<T>>, &str> {
        Ok(self.contents.clone().into_iter().map(|p| Some(p)).collect())
    }

    pub fn min(&self) -> Option<T> {
        self.contents.iter().cloned().min()
    }

    pub fn max(&self) -> Option<T> {
        self.contents.iter().cloned().max()
    }

    pub fn range(&self) -> Option<T> {
        if let (Some(max), Some(min)) = (self.max(), self.min()) {
            Some(max - min)
        } else {
            None
        }

        // if self.contents.length() {
        //     let (min, max) = self.contents.iter().cloned().fold((T::max_value(), T::min_value()), |acc, v| (T::min(acc.0, v.clone()), T::max(acc.1, v)));
        //     Some(max - min)
        // } else {
        //     None
        // }
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
}

#[cfg(test)]
mod tests {
    use crate::collections::numeric::NumericSeq;
    use crate::collections::traits::Collection;

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
}
