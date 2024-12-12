use crate::sequence::member::traits::SequenceMember; // fix
use num_traits::Num;

pub struct NumericMember<T: Clone + Copy + Num> {
    value: T,
}

impl<T: Clone + Copy + Num> NumericMember<T> {
    pub fn new(value: T) -> Box<Self> {
        Box::new(Self { value })
    }
}

impl<T: Clone + Copy + Num> SequenceMember<T> for NumericMember<T> {
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

    fn max(&self) -> Result<T, &str> {
        Ok(self.value)
    }

    fn min(&self) -> Result<T, &str> {
        Ok(self.value)
    }

    fn mean(&self) -> Result<T, &str> {
        Ok(self.value)
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
}

#[cfg(test)]
mod tests {
    use crate::sequence::member::numeric::NumericMember;
    use crate::sequence::member::numeric::SequenceMember;

    use assert_float_eq::assert_f64_near;

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
