/*use crate::algorithms::algorithms;
use crate::sequence::member::chord::ChordMember;
use crate::sequence::member::traits::SequenceMember;
use num_traits::{Num, Zero};
use std::fmt::Debug;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct NumericMember<T: Clone + Copy + Num + PartialOrd + From<i8> + From<i32> + Debug> {
    value: T,
}

impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i8> + From<i32> + Debug> NumericMember<T> {
    pub fn from(value: T) -> Box<Self> {
        Box::new(Self { value })
    }

    pub fn new(value: T) -> Box<Self> {
        Box::new(Self { value })
    }
}

macro_rules! impl_try_from {
    ($source:ty) => {
        impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i8> + From<i32> + Debug>
            TryFrom<$source> for NumericMember<T>
        {
            type Error = String;

            fn try_from(src: $source) -> Result<Self, Self::Error> {
                match src.single_pitch() {
                    Ok(pitch) => Ok(*NumericMember::new(pitch)),
                    Err(x) => Err(x.to_string()),
                }
            }
        }
    };
}

impl_try_from!(ChordMember<T>);

impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i8> + From<i32> + Debug> SequenceMember<T>
    for NumericMember<T>
{
    fn pitches(&self) -> Vec<T> {
        vec![self.value]
    }

    fn num_pitches(&self) -> usize {
        1
    }

    fn single_pitch(&self) -> Result<T, &str> {
        Ok(self.value)
    }

    fn is_silent(&self) -> bool {
        false
    }

    fn max(&self) -> Option<T> {
        Some(self.value)
    }

    fn min(&self) -> Option<T> {
        Some(self.value)
    }

    fn mean(&self) -> Option<T> {
        Some(self.value)
    }

    fn equals(&self, cmp: Self) -> bool {
        self.value == cmp.value
    }

    fn silence(&self) -> Result<Box<Self>, &str> {
        Err("should have one and only one pitch")
    }

    fn map_pitches(&self, f: fn(val: T) -> T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(f(self.value)))
    }

    fn set_pitches(&self, p: Vec<T>) -> Result<Box<Self>, &str> {
        match p.len() {
            1 => Ok(NumericMember::new(p[0])),
            _ => Err("should have one and only one pitch"),
        }
    }

    fn invert(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(p + p - self.value))
    }

    fn transpose(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(self.value + p))
    }

    fn augment(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(p * self.value))
    }

    fn diminish(&self, p: T) -> Result<Box<Self>, String> {
        let f = algorithms::diminish(&p)?;
        Ok(NumericMember::new(f(self.value)))
    }

    fn modulus(&self, p: T) -> Result<Box<Self>, String> {
        let f = algorithms::modulus(&p)?;
        Ok(NumericMember::new(f(self.value)))
    }

    fn trim(&self, a: Option<T>, b: Option<T>) -> Result<Box<Self>, String> {
        let f = algorithms::trim(a.as_ref(), b.as_ref())?;
        Ok(NumericMember::new(f(self.value)))
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::member::numeric::ChordMember;
    use crate::sequence::member::numeric::NumericMember;
    use crate::sequence::member::numeric::SequenceMember;

    use assert_float_eq::assert_f64_near;

    #[test]
    fn from() {
        assert_eq!(NumericMember::from(66), NumericMember::new(66));
    }

    #[test]
    fn try_from() {
        let ch = ChordMember::new(vec![5]);
        let nu = NumericMember::try_from(*ch).unwrap();

        assert_eq!(nu.value, 5);

        let ch = ChordMember::<i32>::new(vec![]);
        assert!(NumericMember::try_from(*ch).is_err());
    }

    #[test]
    fn pitches() {
        assert_eq!(NumericMember::new(5).pitches(), vec![5]);
    }

    #[test]
    fn num_pitches() {
        assert_eq!(NumericMember::new(5).num_pitches(), 1);
    }

    #[test]
    fn single_pitch() {
        assert_eq!(NumericMember::new(5).single_pitch(), Ok(5));
    }

    #[test]
    fn is_silent() {
        assert!(!NumericMember::new(5).is_silent());
    }

    #[test]
    fn max() {
        assert_eq!(NumericMember::new(5).max(), Some(5));
    }

    #[test]
    fn min() {
        assert_eq!(NumericMember::new(5).min(), Some(5));
    }

    #[test]
    fn mean() {
        assert_eq!(NumericMember::new(5).mean(), Some(5));
    }

    #[test]
    fn silence() {
        assert!(NumericMember::new(5).silence().is_err());
    }

    #[test]
    fn map_pitches() {
        assert_eq!(
            NumericMember::new(5).map_pitches(|p| p + 10).unwrap().value,
            15
        );
        assert_f64_near!(
            NumericMember::new(5.6)
                .map_pitches(|p| p * 2.5)
                .unwrap()
                .value,
            14.0
        );
    }

    #[test]
    fn set_pitches() {
        assert!(NumericMember::new(5).set_pitches(vec![]).is_err());
        assert!(NumericMember::new(5).set_pitches(vec![7, 8]).is_err());
        assert_eq!(NumericMember::new(5).set_pitches(vec![6]).unwrap().value, 6);
    }

    #[test]
    fn invert() {
        assert_eq!(NumericMember::new(11).invert(8).unwrap().value, 5);
        assert_f64_near!(NumericMember::new(7.6).invert(1.8).unwrap().value, -4.0);
    }

    #[test]
    fn transpose() {
        assert_eq!(NumericMember::new(11).transpose(8).unwrap().value, 19);
        assert_f64_near!(NumericMember::new(7.6).transpose(1.8).unwrap().value, 9.4);
    }

    #[test]
    fn augment() {
        assert_eq!(NumericMember::new(11).augment(8).unwrap().value, 88);
        assert_f64_near!(NumericMember::new(7.6).augment(1.8).unwrap().value, 13.68);
    }

    #[test]
    fn diminish() {
        assert!(NumericMember::new(11).diminish(0).is_err());
        assert_eq!(NumericMember::new(11).diminish(8).unwrap().value, 1);
        assert_f64_near!(
            NumericMember::new(7.6).diminish(1.8).unwrap().value,
            #[allow(clippy::excessive_precision)]
            4.2222222222222222222
        );
    }

    #[test]
    fn modulus() {
        assert!(NumericMember::new(11).modulus(0).is_err());
        assert_eq!(NumericMember::new(11).modulus(8).unwrap().value, 3);
        // this particularly fails default assertion number of steps
        assert_f64_near!(NumericMember::new(7.6).modulus(1.8).unwrap().value, 0.4, 10);
        assert_f64_near!(NumericMember::new(-7.6).modulus(1.8).unwrap().value, 1.4);
    }

    #[test]
    fn trim() {
        assert_eq!(
            NumericMember::new(11)
                .trim(Some(10), Some(20))
                .unwrap()
                .value,
            11
        );
        assert_eq!(
            NumericMember::new(11).trim(Some(15), None).unwrap().value,
            15
        );
        assert_eq!(
            NumericMember::new(11).trim(None, Some(10)).unwrap().value,
            10
        );
    }
}
*/
