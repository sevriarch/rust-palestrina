use crate::collections::traits::Collection;
use crate::metadata::list::MetadataList;
use crate::sequences::chord::ChordSeq;
use crate::sequences::melody::Melody;
use crate::sequences::note::NoteSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug, PartialEq)]
pub struct NumericSeq<T> {
    contents: Vec<T>,
    metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum NumSeqError {
    InvalidValues,
}

impl<T> TryFrom<Vec<T>> for NumericSeq<T>
where
    T: Copy + Clone + Num + Debug + PartialOrd + Bounded,
{
    type Error = NumSeqError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(NumericSeq::new(what))
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for NumericSeq<T>
where
    T: Copy + Clone + Num + Debug + PartialOrd + Bounded,
{
    type Error = NumSeqError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        let len = what.len();
        let cts: Vec<T> = what
            .into_iter()
            .filter_map(|v| if v.len() == 1 { Some(v[0]) } else { None })
            .collect();

        if cts.len() != len {
            Err(NumSeqError::InvalidValues)
        } else {
            Ok(NumericSeq::new(cts))
        }
    }
}

macro_rules! try_from_seq {
    ($type:ty) => {
        impl<T> TryFrom<$type> for NumericSeq<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = NumSeqError;

            fn try_from(what: $type) -> Result<Self, Self::Error> {
                match what.to_numeric_values() {
                    Ok(vals) => Ok(NumericSeq::new(vals)),
                    Err(_) => Err(NumSeqError::InvalidValues),
                }
            }
        }
    };
}

try_from_seq!(Melody<T>);
try_from_seq!(ChordSeq<T>);
try_from_seq!(NoteSeq<T>);

impl<T: Clone + Num + Debug + PartialOrd + Bounded> Collection<T> for NumericSeq<T> {
    default_collection_methods!(T);
    default_sequence_methods!(T);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>> Sequence<T, T>
    for NumericSeq<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        self.contents.iter_mut().for_each(f);
        self
    }

    fn mutate_pitches_ref<F: Fn(&mut T)>(&mut self, f: F) -> &Self {
        self.mutate_each_ref(f)
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents.clone()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents.clone().into_iter().map(|p| vec![p]).collect()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>, String> {
        Ok(self.contents.clone())
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, String> {
        Ok(self.contents.clone().into_iter().map(|p| Some(p)).collect())
    }
}
// equality methods: equals isSubsetOf isSupersetOf isTransformationOf isTranspositionOf isInversionOf isRetrogradeOf
// isRetrogradeInversionOf hasPeriodicity[Of]

// window replacement methods: replaceIfWindow replaceIfReverseWindow

// setSlice loop repeat dupe dedupe shuffle pad padTo padRight padRightTo withPitch withPitches withPitchesAt
// mapPitches/filterPitches???

// sort chop partitionInPosition groupByInPosition untwine twine combine flatcombine combineMin combineMax combineOr combineAnd
// mapWith filterWith exchangeValuesIf

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::entities::scale::Scale;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::Melody;
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    use assert_float_eq::assert_f64_near;

    #[test]
    fn try_from_vec() {
        assert_eq!(
            NumericSeq::try_from(vec![1, 2, 3]).unwrap(),
            NumericSeq::new(vec![1, 2, 3])
        );
    }

    #[test]
    fn try_from_vec_vec() {
        assert_eq!(
            NumericSeq::try_from(vec![vec![1], vec![2], vec![3]]).unwrap(),
            NumericSeq::new(vec![1, 2, 3])
        );

        assert!(NumericSeq::try_from(vec![Vec::<i32>::new()]).is_err());
        assert!(NumericSeq::try_from(vec![vec![1, 2, 3]]).is_err());
    }

    #[test]
    fn try_from_melody() {
        assert!(NumericSeq::try_from(Melody::<i32>::try_from(vec![vec![]]).unwrap()).is_err());
        assert!(NumericSeq::try_from(Melody::try_from(vec![vec![1, 2, 3]]).unwrap()).is_err());

        assert_eq!(
            NumericSeq::try_from(Melody::try_from(vec![1, 2, 3]).unwrap()),
            Ok(NumericSeq::new(vec![1, 2, 3]))
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert!(NumericSeq::try_from(ChordSeq::<i32>::new(vec![vec![]])).is_err());
        assert!(NumericSeq::try_from(ChordSeq::new(vec![vec![1, 2, 3]])).is_err());

        assert_eq!(
            NumericSeq::try_from(ChordSeq::new(vec![vec![1], vec![2], vec![3]])),
            Ok(NumericSeq::new(vec![1, 2, 3]))
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert!(NumericSeq::try_from(NoteSeq::<i32>::new(vec![None])).is_err());

        assert_eq!(
            NumericSeq::try_from(NoteSeq::new(vec![Some(1), Some(2), Some(3)])),
            Ok(NumericSeq::new(vec![1, 2, 3]))
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_pitches(),
            vec![vec![4], vec![2], vec![5], vec![6], vec![3]]
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            NumericSeq::new(vec![4, 2, 5, 6, 3]).to_flat_pitches(),
            vec![4, 2, 5, 6, 3]
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
    fn min_value() {
        assert!(NumericSeq::<i64>::new(vec![]).min_value().is_none());
        assert_eq!(NumericSeq::new(vec![4, 2, 5, 6, 3]).min_value(), Some(2));
        assert_eq!(
            NumericSeq::new(vec![4.1, 2.8, 5.4, 6.3, 3.0]).min_value(),
            Some(2.8)
        );
    }

    #[test]
    fn max_value() {
        assert!(NumericSeq::<i64>::new(vec![]).max_value().is_none());
        assert_eq!(NumericSeq::new(vec![4, 2, 5, 6, 3]).max_value(), Some(6));
        assert_eq!(
            NumericSeq::new(vec![4.1, 2.8, 5.4, 6.3, 3.0]).max_value(),
            Some(6.3)
        );
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
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).transpose(2).unwrap(),
            NumericSeq::new(vec![3, 8, 6])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .transpose(-1.8)
                .unwrap(),
            NumericSeq::new(vec![-0.1, 1.6, 4.5])
        );
    }

    #[test]
    fn transpose_to_min() {
        assert_eq!(
            NumericSeq::<i32>::new(vec![]).transpose_to_min(44).unwrap(),
            NumericSeq::<i32>::new(vec![])
        );
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).transpose_to_min(2).unwrap(),
            NumericSeq::new(vec![2, 7, 5])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .transpose_to_min(-1.8)
                .unwrap(),
            NumericSeq::new(vec![-1.8, -0.1, 2.8])
        );
    }

    #[test]
    fn transpose_to_max() {
        assert_eq!(
            NumericSeq::new(vec![]).transpose_to_max(44).unwrap(),
            NumericSeq::new(vec![])
        );
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).transpose_to_max(2).unwrap(),
            NumericSeq::new(vec![-3, 2, 0])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .transpose_to_max(-1.8)
                .unwrap(),
            NumericSeq::new(vec![-6.4, -4.7, -1.8])
        );
    }

    #[test]
    fn invert() {
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).invert(2).unwrap(),
            NumericSeq::new(vec![3, -2, 0])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3]).invert(-1.8).unwrap(),
            NumericSeq::new(vec![-5.3, -7.0, -9.9])
        );
    }

    #[test]
    fn augment() {
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).augment(2).unwrap(),
            NumericSeq::new(vec![2, 12, 8])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3]).augment(2.0).unwrap(),
            NumericSeq::new(vec![3.4, 6.8, 12.6])
        );
    }

    #[test]
    fn diminish() {
        assert!(NumericSeq::new(vec![1, 6, 4]).diminish(0).is_err());
        assert_eq!(
            NumericSeq::new(vec![1, 6, 4]).diminish(2).unwrap(),
            NumericSeq::new(vec![0, 3, 2])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3]).diminish(2.0).unwrap(),
            NumericSeq::new(vec![0.85, 1.7, 3.15])
        );
    }

    #[test]
    fn modulus() {
        assert!(NumericSeq::new(vec![-1, 6, 4]).modulus(0).is_err());
        assert_eq!(
            NumericSeq::new(vec![-1, 6, 4]).modulus(3).unwrap(),
            NumericSeq::new(vec![2, 0, 1])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![-1.7, 3.4, 6.3]).modulus(2.0).unwrap(),
            NumericSeq::new(vec![0.3, 1.4, 0.3])
        );
    }

    #[test]
    fn trim() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 5, 6, 4]).trim(2, 5).unwrap(),
            NumericSeq::new(vec![2, 2, 5, 5, 4])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3]).trim(2.0, 5.0).unwrap(),
            NumericSeq::new(vec![2.0, 3.4, 5.0])
        );
    }

    #[test]
    fn trim_min() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 5, 6, 4]).trim_min(2).unwrap(),
            NumericSeq::new(vec![2, 2, 5, 6, 4])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![2.0, 3.4, 6.3]).trim_min(2.0).unwrap(),
            NumericSeq::new(vec![2.0, 3.4, 6.3])
        );
    }

    #[test]
    fn trim_max() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 5, 6, 4]).trim_max(5).unwrap(),
            NumericSeq::new(vec![1, 2, 5, 5, 4])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3]).trim_max(5.0).unwrap(),
            NumericSeq::new(vec![1.7, 3.4, 5.0])
        );
    }

    #[test]
    fn bounce() {
        assert_eq!(
            NumericSeq::new(vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
            ])
            .bounce(7, 10)
            .unwrap(),
            NumericSeq::new(vec![
                8, 7, 8, 9, 10, 9, 8, 7, 8, 9, 10, 9, 8, 7, 8, 9, 10, 9
            ])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .bounce(2.0, 3.0)
                .unwrap(),
            NumericSeq::new(vec![2.3, 2.6, 2.3])
        );
    }

    #[test]
    fn bounce_min() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 5, 6, 4]).bounce_min(2).unwrap(),
            NumericSeq::new(vec![3, 2, 5, 6, 4])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .bounce_min(2.0)
                .unwrap(),
            NumericSeq::new(vec![2.3, 3.4, 6.3])
        );
    }

    #[test]
    fn bounce_max() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 5, 6, 4]).bounce_max(5).unwrap(),
            NumericSeq::new(vec![1, 2, 5, 4, 4])
        );

        assert_contents_f64_near!(
            NumericSeq::new(vec![1.7, 3.4, 6.3])
                .bounce_max(5.0)
                .unwrap(),
            NumericSeq::new(vec![1.7, 3.4, 3.7])
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
            NumericSeq::new(vec![
                26, 28, 30, 31, 33, 35, 36, 38, 40, 42, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60,
                62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91, 93
            ])
        );

        assert_eq!(
            NumericSeq::new(vec![-1, 0, 1]).scale(lydian, 20).unwrap(),
            NumericSeq::new(vec![19, 20, 22])
        );
    }

    #[test]
    fn filter_in_position() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 3, 4, 5])
                .filter_in_position(|v| v % 2 == 0, 8)
                .unwrap(),
            NumericSeq::new(vec![8, 2, 8, 4, 8])
        );
    }

    #[test]
    fn flat_map_windows() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 3, 4, 5])
                .flat_map_windows(2, 1, |mut w| {
                    w.reverse();
                    w
                })
                .unwrap(),
            NumericSeq::new(vec![2, 1, 3, 2, 4, 3, 5, 4])
        );

        assert_eq!(
            NumericSeq::new(vec![1, 2, 3, 4, 5])
                .flat_map_windows(3, 2, |mut w| {
                    w.reverse();
                    w
                })
                .unwrap(),
            NumericSeq::new(vec![3, 2, 1, 5, 4, 3])
        );
    }

    #[test]
    fn filter_windows() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 3, 4, 5])
                .filter_windows(2, 1, |w| w[0] > 1)
                .unwrap(),
            NumericSeq::new(vec![2, 3, 3, 4, 4, 5])
        );

        assert_eq!(
            NumericSeq::new(vec![1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
                .filter_windows(3, 2, |w| w[0] > 2)
                .unwrap(),
            NumericSeq::new(vec![3, 4, 5, 5, 6, 5, 5, 4, 3, 3, 2, 1])
        );
    }

    #[test]
    fn pad() {
        assert_eq!(
            NumericSeq::new(vec![1, 2, 3]).pad(4, 1),
            &NumericSeq::new(vec![4, 1, 2, 3])
        );

        assert_eq!(
            NumericSeq::new(vec![1, 2, 3]).pad(2, 4),
            &NumericSeq::new(vec![2, 2, 2, 2, 1, 2, 3])
        );
    }

    #[test]
    fn test_chaining() {
        fn chained_methods() -> Result<NumericSeq<i32>, String> {
            let ret = NumericSeq::new(vec![1, 2, 3])
                .augment(3)?
                .transpose(1)?
                .invert(13)?;

            Ok(ret)
        }

        assert_eq!(
            chained_methods().unwrap(),
            NumericSeq::new(vec![22, 19, 16])
        )
    }
}
