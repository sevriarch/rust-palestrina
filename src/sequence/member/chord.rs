use crate::sequence::member::numeric::NumericMember;
use crate::sequence::member::traits::SequenceMember; // fix
use num_traits::{Num, Zero};
use std::cmp::Ord;

pub struct ChordMember<T: Clone + Copy + Num + Zero + Ord + From<i32>> {
    value: Vec<T>,
}

impl<T: Clone + Copy + Num + Zero + Ord + From<i32>> ChordMember<T> {
    fn new(mut value: Vec<T>) -> Box<Self> {
        value.sort();
        Box::new(Self { value })
    }
}

impl<T: Clone + Copy + Num + Zero + Ord + From<i32>> From<NumericMember<T>> for ChordMember<T> {
    fn from(src: NumericMember<T>) -> Self {
        *ChordMember::new(src.pitches())
    }
}

impl<T: Clone + Copy + Num + Zero + Ord + From<i32>> SequenceMember<T> for ChordMember<T> {
    fn pitches(&self) -> Vec<T> {
        self.value.clone()
    }

    fn num_pitches(&self) -> usize {
        self.value.len()
    }

    fn single_pitch(&self) -> Result<T, &str> {
        match self.value.len() {
            1 => Ok(self.value[0]),
            _ => Err("can only have one value"),
        }
    }

    fn is_silent(&self) -> bool {
        self.value.is_empty()
    }

    fn max(&self) -> Option<T> {
        self.value.iter().max().copied()
    }

    fn min(&self) -> Option<T> {
        self.value.iter().min().copied()
    }

    fn mean(&self) -> Option<T> {
        let mut iter = self.value.iter();
        let first = iter.next()?;

        Some(iter.fold(*first, |acc, x| acc + *x) / T::from(self.value.len() as i32))
        /*
        match self.value.is_empty() {
            true => None,
            false => Some(self.value.iter().fold(T::zero(), |acc, x| acc + *x) / T::from(self.value.len() as i32)),
        }
        */
    }

    fn equals(&self, cmp: Self) -> bool {
        self.value == cmp.value
    }

    fn silence(&self) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(vec![]))
    }

    fn map_pitches(&self, f: fn(val: T) -> T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(f).collect(),
        ))
    }

    fn set_pitches(&self, p: Vec<T>) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(p))
    }

    fn invert(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(|v| p + p - v).collect(),
        ))
    }

    fn transpose(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(|v| p + v).collect(),
        ))
    }

    fn augment(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(|v| p * v).collect(),
        ))
    }

    fn diminish(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(|v| v / p).collect(),
        ))
    }

    fn modulus(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(
            self.value.clone().into_iter().map(|v| v % p).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::member::chord::ChordMember;
    use crate::sequence::member::numeric::NumericMember;
    use crate::sequence::member::traits::SequenceMember;

    //use assert_float_eq::assert_f64_near;

    #[test]
    fn pitches() {
        assert_eq!(ChordMember::new(vec![5, 7]).pitches(), vec![5, 7]);
    }

    #[test]
    fn num_pitches() {
        assert_eq!(ChordMember::new(vec![5, 7]).num_pitches(), 2);
    }

    #[test]
    fn single_pitch() {
        assert!(ChordMember::<i64>::new(vec![]).single_pitch().is_err());
        assert!(ChordMember::new(vec![5, 7]).single_pitch().is_err());
        assert_eq!(ChordMember::new(vec![7]).single_pitch(), Ok(7));
    }

    #[test]
    fn is_silent() {
        assert!(ChordMember::<i64>::new(vec![]).is_silent());
        assert!(!ChordMember::new(vec![5]).is_silent());
    }

    #[test]
    fn max() {
        assert!(ChordMember::<i64>::new(vec![]).max().is_none());
        assert_eq!(ChordMember::new(vec![5]).max(), Some(5));
        assert_eq!(ChordMember::new(vec![5, 7]).max(), Some(7));
    }

    #[test]
    fn min() {
        assert!(ChordMember::<i64>::new(vec![]).min().is_none());
        assert_eq!(ChordMember::new(vec![5]).min(), Some(5));
        assert_eq!(ChordMember::new(vec![5, 7]).min(), Some(5));
    }

    #[test]
    fn mean() {
        assert!(ChordMember::<i64>::new(vec![]).mean().is_none());
        assert_eq!(ChordMember::new(vec![5]).mean(), Some(5));
        assert_eq!(ChordMember::new(vec![5, 7]).mean(), Some(6));
    }

    #[test]
    fn equals() {
        assert!(ChordMember::new(vec![5, 7]).equals(*ChordMember::new(vec![7, 5])));
        assert!(!ChordMember::new(vec![5, 7, 0]).equals(*ChordMember::new(vec![7, 5])));
    }

    #[test]
    fn from_numeric() {
        assert_eq!(ChordMember::from(*NumericMember::new(7)).pitches(), vec![7]);
    }

    #[test]
    fn invert() {
        assert_eq!(ChordMember::new(vec![2]).invert(8).unwrap().value, vec![14]);
        //assert_f64_near!(ChordMember::new(vec![7.6,6.8]).invert(1.8).unwrap().value, vec![-4.0,-3.2]);//fails bc Ord trait no work for f32
    }
}
