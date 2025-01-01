use crate::collections::traits::Collection;
use crate::default_collection_methods;
use crate::metadata::list::MetadataList;
use crate::sequences::melody::Melody;
use crate::sequences::traits::Sequence;
use num_traits::{Bounded, Num};
use std::convert::From;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct Score<T>
where
    T: Clone + Copy + Debug,
{
    contents: Vec<Melody<T>>,
    metadata: MetadataList,
}

impl<T> From<Vec<Melody<T>>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    fn from(m: Vec<Melody<T>>) -> Self {
        Self::new(m)
    }
}

impl<T> From<Melody<T>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    fn from(m: Melody<T>) -> Self {
        Self::new(vec![m])
    }
}

impl<T: Clone + Copy + Debug> Collection<Melody<T>> for Score<T>
where
    T: Clone + Copy + Debug,
{
    default_collection_methods!(Melody<T>);

    fn new(m: Vec<Melody<T>>) -> Self {
        Self {
            contents: m,
            metadata: MetadataList::default(),
        }
    }

    fn construct(&self, contents: Vec<Melody<T>>) -> Self {
        Self {
            contents,
            metadata: self.metadata.clone(),
        }
    }
}

impl<T> Score<T>
where
    T: Clone + Copy + Debug + PartialOrd + Bounded + From<i32> + Num + Sum,
{
    pub fn pitch_range(&self) -> Option<(T, T)> {
        let min = self
            .contents
            .iter()
            .filter_map(|m| m.min_value())
            .min_by(|a, b| a.partial_cmp(b).unwrap())?;
        let max = self
            .contents
            .iter()
            .filter_map(|m| m.max_value())
            .max_by(|a, b| a.partial_cmp(b).unwrap())?;

        Some((min, max))
    }

    pub fn volume_range(&self) -> Option<(u8, u8)> {
        let min = self
            .contents
            .iter()
            .filter_map(|m| m.min_volume())
            .min_by(|a, b| a.partial_cmp(b).unwrap())?;
        let max = self
            .contents
            .iter()
            .filter_map(|m| m.max_volume())
            .max_by(|a, b| a.partial_cmp(b).unwrap())?;

        Some((min, max))
    }

    pub fn last_tick(&self) -> u32 {
        let metalast = self.metadata.last_tick(0).unwrap_or(0);

        self.contents.iter().fold(metalast, |max, m| {
            let ret = m.last_tick().unwrap_or(0);

            if ret > max {
                ret
            } else {
                max
            }
        })
    }

    pub fn with_all_ticks_exact(&self) -> &Self {
        todo!();
    }

    pub fn to_canvas(&self) -> String {
        todo!();
    }

    pub fn to_midi_bytes(&self) -> &[u8] {
        todo!();
    }

    pub fn to_data_uri(&self) -> String {
        todo!();
    }

    pub fn to_hash(&self) -> String {
        todo!();
    }

    pub fn expect_hash(&self) -> Result<(), String> {
        todo!();
    }

    pub fn write_canvas(&self) -> Result<(), String> {
        todo!();
    }

    pub fn write_midi(&self) -> Result<(), String> {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::timing::DurationalEventTiming;
    use crate::sequences::melody::MelodyMember;

    #[test]
    fn pitch_range() {
        assert!(Score::new(vec![Melody::<i32>::new(vec![])])
            .pitch_range()
            .is_none());

        assert_eq!(
            Score::new(vec![
                Melody::try_from(vec![1, 2, 3, 4, 5]).unwrap(),
                Melody::<i32>::new(vec![]),
                Melody::try_from(vec![6, 7, 8]).unwrap(),
            ])
            .pitch_range(),
            Some((1, 8))
        );
    }

    macro_rules! mmv {
        ($p:expr, $v:expr) => {
            MelodyMember {
                values: $p,
                timing: DurationalEventTiming::default(),
                volume: $v,
                before: MetadataList::new(vec![]),
            }
        };
    }

    #[test]
    fn volume_range() {
        assert!(Score::new(vec![Melody::<i32>::new(vec![])])
            .volume_range()
            .is_none());

        assert_eq!(
            Score::new(vec![
                Melody::try_from(vec![mmv!(vec![1], 40), mmv!(vec![], 80), mmv!(vec![2], 60)])
                    .unwrap(),
                Melody::<i32>::new(vec![]),
                Melody::try_from(vec![mmv!(vec![1], 50), mmv!(vec![], 30), mmv!(vec![2], 70)])
                    .unwrap(),
            ])
            .volume_range(),
            Some((40, 70))
        );
    }
}
