use crate::collections::traits::Collection;
use crate::default_collection_methods;
use crate::metadata::{MetadataList, PushMetadata};
use crate::ops::pitch::Pitch;
use crate::sequences::melody::Melody;
use crate::sequences::traits::Sequence;
use anyhow::Result;
use num_traits::{Bounded, FromPrimitive, Num};
use std::convert::From;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

#[derive(Clone, Debug)]
pub struct Score<T>
where
    T: Clone + Copy + Debug,
{
    pub contents: Vec<Melody<T>>,
    pub metadata: MetadataList,
    pub ticks_per_quarter: u32,
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
            ticks_per_quarter: 192,
        }
    }

    fn construct(&self, contents: Vec<Melody<T>>) -> Self {
        Self {
            contents,
            metadata: self.metadata.clone(),
            ticks_per_quarter: 192,
        }
    }
}

impl<T> Score<T>
where
    T: Pitch<T>
        + Clone
        + Copy
        + Debug
        + FromPrimitive
        + PartialOrd
        + Bounded
        + FromPrimitive
        + From<i32>
        + Num
        + Sum
        + AddAssign,
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

    pub fn last_tick(&self) -> i32 {
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

    pub fn with_ticks_per_quarter(mut self, ticks: u32) -> Self {
        self.ticks_per_quarter = ticks;
        self
    }

    pub fn ticks_per_quarter(&self) -> u32 {
        self.ticks_per_quarter
    }

    pub fn with_tempo<TTempo: Into<f32>>(mut self, tempo: TTempo) -> Result<Self> {
        self.metadata = self.metadata.with_metadata(("tempo", tempo.into()))?;
        Ok(self)
    }

    pub fn with_time_signature(mut self, ts: &str) -> Result<Self> {
        self.metadata = self.metadata.with_metadata(("time-signature", ts))?;
        Ok(self)
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
    use crate::entities::timing::{DurationalEventTiming, EventTiming};
    use crate::metadata::{Metadata, MetadataData};
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

    #[test]
    fn with_ticks_per_quarter() {
        let sc = Score::new(vec![Melody::<i32>::new(vec![])]);

        assert_eq!(sc.ticks_per_quarter(), 192);
        assert_eq!(sc.with_ticks_per_quarter(64).ticks_per_quarter(), 64);
    }

    #[test]
    fn with_tempo() {
        let sc = Score::new(vec![Melody::<i32>::new(vec![])])
            .with_tempo(144_i16)
            .unwrap();

        assert_eq!(
            sc.metadata,
            MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Tempo(144.0),
                    timing: EventTiming::default()
                }]
            }
        );
    }

    #[test]
    fn with_time_signature() {
        let sc = Score::new(vec![Melody::<i32>::new(vec![])])
            .with_time_signature("4/4")
            .unwrap();

        assert_eq!(
            sc.metadata,
            MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::TimeSignature((4, 4)),
                    timing: EventTiming::default()
                }]
            }
        );
    }
}
