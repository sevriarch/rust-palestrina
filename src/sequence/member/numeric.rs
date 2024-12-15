use crate::mutators;
use crate::sequence::member::chord::ChordMember;
use crate::sequence::member::traits::SequenceMember;
use num_traits::{Num, Zero};

pub struct NumericMember<T: Clone + Copy + Num + PartialOrd + From<i32>> {
    value: T,
}

impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i32>> NumericMember<T> {
    pub fn new(value: T) -> Box<Self> {
        Box::new(Self { value })
    }
}

macro_rules! impl_try_from {
    ($source:ty) => {
        impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i32>> TryFrom<$source>
            for NumericMember<T>
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

impl<T: Clone + Copy + Num + Zero + PartialOrd + From<i32>> SequenceMember<T> for NumericMember<T> {
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

    fn diminish(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(self.value / p))
    }

    fn modulus(&self, p: T) -> Result<Box<Self>, &str> {
        Ok(NumericMember::new(self.value % p))
    }

    fn trim(&self, a: Option<T>, b: Option<T>) -> Result<Box<Self>, &str> {
        let f = mutators::trim(a.as_ref(), b.as_ref())?;

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
    fn invert() {
        assert_eq!(NumericMember::new(11).invert(8).unwrap().value, 5);
        assert_f64_near!(NumericMember::new(7.6).invert(1.8).unwrap().value, -4.0);
    }
}
