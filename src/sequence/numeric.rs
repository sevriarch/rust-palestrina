// Traits common to all Sequences
use crate::collections::traits::Collection;
use crate::sequence::member::numeric::NumericMember;
use num_traits::Num;
use std::fmt::Debug;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct NumericSequence<T: Clone + Copy + Debug + Num + PartialOrd + From<i8> + From<i32>> {
    contents: Vec<NumericMember<T>>,
}

impl<T: Clone + Copy + Debug + Num + PartialOrd + From<i8> + From<i32>> NumericSequence<T> {
    pub fn from(contents: Vec<T>) -> Box<Self> {
        Box::new(Self {
            contents: contents
                .into_iter()
                .map(move |c| {
                    let mem = *NumericMember::from(c).as_ref();

                    mem
                })
                .collect(),
        })
    }
}

impl<T: Clone + Copy + Debug + Num + PartialOrd + From<i8> + From<i32>> Collection<NumericMember<T>>
    for NumericSequence<T>
{
    fn new(contents: Vec<NumericMember<T>>) -> Self {
        Self { contents }
    }

    fn cts(&self) -> Vec<NumericMember<T>> {
        self.contents.clone()
    }

    fn length(&self) -> usize {
        self.contents.len()
    }

    fn construct(&self, contents: Vec<NumericMember<T>>) -> Box<Self> {
        let mut newseq = self.clone();
        newseq.contents = contents;
        Box::new(newseq)
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::sequence::member::numeric::NumericMember;
    use crate::sequence::member::traits::SequenceMember;
    use crate::sequence::numeric::NumericSequence;

    #[test]
    fn filter() {
        let coll = NumericSequence::from(vec![0, 2, 3, 4, 5]);
        let new = coll.filter(|v| v.single_pitch().unwrap() % 2 == 0);

        assert_eq!(
            new.contents,
            vec![
                *NumericMember::new(0),
                *NumericMember::new(2),
                *NumericMember::new(4)
            ]
        );
    }

    #[test]
    fn map() {
        let coll = NumericSequence::from(vec![0, 2, 3, 4, 5]);
        let new = coll.map(|v| *v.augment(v.single_pitch().unwrap()).unwrap());

        assert_eq!(
            new.contents,
            vec![
                *NumericMember::new(0),
                *NumericMember::new(4),
                *NumericMember::new(9),
                *NumericMember::new(16),
                *NumericMember::new(25)
            ]
        );
    }

    /*
    #[test]
    fn add() {
        let c1 = Sequence::new(vec![1, 5, 6]);
        let c2 = Sequence::new(vec![2, 7, 8, 9]);

        let c3 = Sequence::new(vec![1, 5, 6, 2, 7, 8, 9]);

        assert_eq!((c1 + c2).contents, c3.contents);
    }
    */
}
