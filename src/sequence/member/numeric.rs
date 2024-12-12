use crate::sequence::member::member::SequenceMember; // fix

pub struct NumericMember<T> {
    value: T,
}

impl SequenceMember for NumericMember {
    fn pitches(&self) -> Vec<T> {
        vec![self.value]
    }

    fn num_pitches(&self) -> usize {
        1
    }

    fn single_pitch(&self) -> Option<T> {
        Some(self.value)
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
        this.value == cmp.value
    }

    fn map_pitches(&self, f: fn(val: T) -> T) -> self {
        NumericMember::new(f(self.value))
    }

    fn set_pitches(&self, p: Vec<T>) -> Option<self> {
    }
}