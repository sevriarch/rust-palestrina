use crate::sequence::member::traits::SequenceMember; // fix
use crate::sequence::member::numeric::NumericMember;
use num_traits::Num;

pub struct ChordMember<T: Clone + Copy + Num> {
    value: Vec<T>,
}

impl<T: Clone + Copy + Num> ChordMember<T> {
    fn new(value: Vec<T>) -> Box<Self> {
        Box::new(Self { value })
    }
}

impl<T: Clone + Copy + Num> From<NumericMember<T>> for ChordMember<T> {
    fn from(src: NumericMember<T>) -> Self {
        *ChordMember::new(src.pitches())
    }
}

impl<T: Clone + Copy + Num> SequenceMember<T> for ChordMember<T> {
    fn pitches(&self) -> Vec<T> {
        self.value.clone()
    }

    fn num_pitches(&self) -> usize {
        1
    }

    fn single_pitch(&self) -> Result<T, &str> {
        match self.value.len() {
            1 => Ok(self.value[0]),
            _ => Err("can only have one value"),
        }
    }

    fn is_silent(&self) -> bool {
        false
    }

    fn max(&self) -> Result<T, &str> {
        todo!()
    }

    fn min(&self) -> Result<T, &str> {
        todo!()
    }

    fn mean(&self) -> Result<T, &str> {
        todo!()
    }

    fn equals(&self, cmp: Self) -> bool {
        self.value == cmp.value
    }

    fn silence(&self) -> Result<Box<Self>, &str> {
        Err("should have one and only one pitch")
    }

    fn map_pitches(&self, f: fn(val: T) -> T) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(self.value.clone().into_iter().map(f).collect()))
    }

    fn set_pitches(&self, p: Vec<T>) -> Result<Box<Self>, &str> {
        Ok(ChordMember::new(p))
    }

    fn invert(&self, p: T) -> Result<Box<Self>, &str> {
        todo!()
    }

    fn transpose(&self, p: T) -> Result<Box<Self>, &str> {
        todo!()
    }

    fn augment(&self, p: T) -> Result<Box<Self>, &str> {
        todo!()
    }

    fn diminish(&self, p: T) -> Result<Box<Self>, &str> {
        todo!()
    }

    fn modulus(&self, p: T) -> Result<Box<Self>, &str> {
        todo!()
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
        assert_eq!(ChordMember::new(vec![5,7]).pitches(), vec![5,7]);
    }

    #[test]
    fn from_numeric() {
        assert_eq!(ChordMember::from(*NumericMember::new(7)).pitches(), vec![7]);
    }

    /*
    #[test]
    fn invert() {
        assert_eq!(ChordMember::new(ve11).invert(8).unwrap().value, 5);
        assert_f64_near!(ChordMember::new(7.6).invert(1.8).unwrap().value, -4.0);
    }
    */
}
