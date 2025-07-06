use crate::algorithms;
use crate::collections::traits::Collection;
use crate::entities::timing::{DurationalEventTiming, Timing};
use crate::metadata::{Metadata, MetadataList};
use crate::ops::pitch::{AugDim, Pitch};
use crate::sequences::chord::ChordSeq;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use anyhow::{anyhow, Result};
use num_traits::{Bounded, FromPrimitive, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

pub const DEFAULT_VOLUME: u8 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct MelodyMember<T> {
    pub values: Vec<T>,
    pub timing: DurationalEventTiming,
    pub volume: u8,
    pub before: MetadataList,
}

#[macro_export]
/// A macro for the most common cases of creating a member of a Melody.
///
/// Zero arguments: No notes, default volume and duration.
/// One numeric argument: One note, default volume and duration.
/// One vec argument: The notes in this vec, default volume and duration.
/// Multiple notes encased in [[]], eg: [[1,2,3]]: These notes, default volume and duration.
/// Two arguments: As for one argument, but the second argument supplies volume.
/// Three arguments: As for two arguments, but the third argument supplies duration.
macro_rules! melody_member {
    () => {
        MelodyMember::default()
    };

    ([$v:expr]) => {
        MelodyMember::from($v.to_vec())
    };

    ($v:expr) => {
        MelodyMember::from($v)
    };

    ([$v:expr], $vol:expr) => {
        MelodyMember {
            values: $v.to_vec(),
            volume: $vol,
            ..Default::default()
        }
    };

    ($v:expr, $vol:expr) => {
        MelodyMember {
            values: vec![$v],
            volume: $vol,
            ..Default::default()
        }
    };

    ([$v:expr], $vol:expr, $dur:expr) => {
        MelodyMember {
            values: $v.to_vec(),
            volume: $vol,
            timing: DurationalEventTiming::from_duration($dur),
            ..Default::default()
        }
    };

    ($v:expr, $vol:expr, $dur:expr) => {
        MelodyMember {
            values: vec![$v],
            volume: $vol,
            timing: DurationalEventTiming::from_duration($dur),
            ..Default::default()
        }
    };
}

impl<T> Default for MelodyMember<T> {
    fn default() -> Self {
        MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            volume: DEFAULT_VOLUME,
            before: MetadataList::new(vec![]),
        }
    }
}

impl<T> From<T> for MelodyMember<T>
where
    T: Copy + Clone + Num,
{
    fn from(what: T) -> Self {
        MelodyMember {
            values: vec![what],
            ..Default::default()
        }
    }
}

impl<T> From<Option<T>> for MelodyMember<T>
where
    T: Copy + Clone + Num,
{
    fn from(what: Option<T>) -> Self {
        MelodyMember {
            values: what.map_or_else(Vec::new, |v| vec![v]),
            ..Default::default()
        }
    }
}

impl<T> From<Vec<T>> for MelodyMember<T>
where
    T: Copy + Clone + Num,
{
    fn from(what: Vec<T>) -> Self {
        MelodyMember {
            values: what,
            ..Default::default()
        }
    }
}

macro_rules! impl_fns_for_melody_member {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(self, n: $ty) -> Self {
            self.values.iter_mut().for_each(|p| { *p = *p.$fn(n); });
            self
        }
    )*)
}

impl<T> Pitch<T> for &mut MelodyMember<T>
where
    T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum,
{
    impl_fns_for_melody_member!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn set_pitches(self, p: Vec<T>) -> Result<Self> {
        self.values = p;
        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(self, f: MapT) -> Self {
        self.values.iter_mut().for_each(|p| *p = f(p));
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(self, f: FilterT) -> Result<Self> {
        self.values.retain(f);
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(self, f: MapT) -> Result<Self> {
        self.values = self
            .values
            .clone()
            .into_iter()
            .filter_map(|v| f(&v))
            .collect();
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.values.iter_mut().for_each(|p| {
            p.augment_pitch(n);
        });
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(self, n: AT) -> Self {
        self.values.iter_mut().for_each(|p| {
            p.diminish_pitch(n);
        });
        self
    }

    fn trim(self, first: T, last: T) -> Self {
        self.values.iter_mut().for_each(|p| {
            p.trim(first, last);
        });
        self
    }

    fn bounce(self, first: T, last: T) -> Self {
        self.values.iter_mut().for_each(|p| {
            p.bounce(first, last);
        });
        self
    }

    fn is_silent(self) -> bool {
        self.values.is_empty() || self.volume == 0
    }
}

impl<T> MelodyMember<T>
where
    T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum,
{
    pub fn new(values: Vec<T>) -> Self {
        Self {
            values,
            ..Default::default()
        }
    }

    pub fn with_event(&mut self, e: &Metadata) -> &mut Self {
        self.before.append(e.clone());

        self
    }

    pub fn last_tick(&self, curr: u32) -> Result<u32> {
        let mut max = 0;

        if !self.values.is_empty() {
            let end_of_note = self.timing.end_tick(curr)?;

            if end_of_note > max {
                max = end_of_note;
            }
        }

        if !self.before.contents.is_empty() {
            let last_metadata = self.before.last_tick(self.timing.start_tick(curr)?)?;

            if last_metadata > max {
                max = last_metadata;
            }
        }

        Ok(max)
    }

    pub fn with_exact_tick(&mut self, d: u32) -> &mut Self {
        self.timing.with_exact_tick(d);
        self
    }

    pub fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> &mut Self {
        self.before.contents.iter_mut().for_each(|m| {
            m.mutate_exact_tick(&f);
        });

        self.timing.mutate_exact_tick(f);
        self
    }

    pub fn with_offset(&mut self, d: i32) -> &mut Self {
        self.timing.with_offset(d);
        self
    }

    pub fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> &mut Self {
        self.before.contents.iter_mut().for_each(|m| {
            m.mutate_offset(&f);
        });

        self.timing.mutate_offset(&f);
        self
    }

    pub fn with_duration(&mut self, d: u32) -> &mut Self {
        self.timing.with_duration(d);
        self
    }

    pub fn mutate_duration(&mut self, f: impl Fn(&mut u32)) -> &mut Self {
        self.timing.mutate_duration(f);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Melody<T> {
    pub contents: Vec<MelodyMember<T>>,
    pub metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum MelodyError {
    InvalidValues,
}

#[macro_export]
macro_rules! melody {
    ($([$($x:expr),*]),*) => (
        Melody::new(vec![ $(MelodyMember::from(vec![ $($x),* ])),* ])
    );

    ($($x:expr),*) => (
        Melody::new(vec![ $(MelodyMember::from(vec![ $x ])),* ])
    );
}

macro_rules! try_from_vec {
    (for $($t:ty)*) => ($(
        impl<T> TryFrom<$t> for Melody<T>
        where
            T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + AddAssign,
        {
            type Error = anyhow::Error;

            fn try_from(what: $t) -> Result<Self> {
                Ok(Self::new(what.into_iter().map(MelodyMember::from).collect()))
            }
        }
    )*)
}

try_from_vec!(for Vec<T> Vec<Option<T>> Vec<Vec<T>>);

impl<T> TryFrom<Vec<MelodyMember<T>>> for Melody<T>
where
    T: Pitch<T>
        + Clone
        + Copy
        + Num
        + Debug
        + FromPrimitive
        + PartialOrd
        + Bounded
        + Sum
        + AddAssign,
{
    type Error = anyhow::Error;

    fn try_from(what: Vec<MelodyMember<T>>) -> Result<Self> {
        Ok(Self::new(what))
    }
}

macro_rules! try_from_seq {
    (for $($t:ty)*) => ($(
        impl<T> TryFrom<$t> for Melody<T>
        where
            T: Pitch<T> + Copy + Num + Debug + FromPrimitive + PartialOrd + Bounded + Sum,
        {
            type Error = anyhow::Error;

            fn try_from(what: $t) -> Result<Self> {
                Ok(Self {
                    contents: what
                        .to_pitches()
                        .into_iter()
                        .map(MelodyMember::from)
                        .collect(),
                    metadata: what.metadata,
                })
            }
        }
    )*)
}

try_from_seq!(for NumericSeq<T> NoteSeq<T> ChordSeq<T>);

impl<T> Collection<MelodyMember<T>> for Melody<T>
where
    T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum,
{
    default_collection_methods!(MelodyMember<T>);
    default_sequence_methods!(MelodyMember<T>);
}

impl<T> Sequence<MelodyMember<T>, T> for Melody<T>
where
    T: Pitch<T>
        + Clone
        + Copy
        + Num
        + Debug
        + FromPrimitive
        + PartialOrd
        + AddAssign
        + Bounded
        + Sum,
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        self.contents.iter_mut().for_each(|m| {
            for p in m.values.iter_mut() {
                f(p)
            }
        });
        self
    }

    fn to_flat_pitches(&self) -> Vec<T> {
        self.contents
            .iter()
            .flat_map(|v| v.values.iter())
            .copied()
            .collect()
    }

    fn to_pitches(&self) -> Vec<Vec<T>> {
        self.contents.iter().map(|v| v.values.clone()).collect()
    }

    fn to_numeric_values(&self) -> Result<Vec<T>> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                1 => Ok(v.values[0]),
                _ => Err(anyhow!("must contain only one value")),
            })
            .collect()
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                0 => Ok(None),
                1 => Ok(Some(v.values[0])),
                _ => Err(anyhow!("must contain zero or one values")),
            })
            .collect()
    }

    fn map_pitch_enumerated<MapT: Fn((usize, &T)) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().enumerate().for_each(|(i, m)| {
            m.map_pitch(|v| f((i, v)));
        });
        self
    }

    fn filter_pitch_enumerated<FilterT: Fn((usize, &T)) -> bool>(
        mut self,
        f: FilterT,
    ) -> Result<Self> {
        for (i, m) in self.contents.iter_mut().enumerate() {
            m.filter_pitch(|v| f((i, v)))?;
        }
        Ok(self)
    }

    fn filter_map_pitch_enumerated<MapT: Fn((usize, &T)) -> Option<T>>(
        mut self,
        f: MapT,
    ) -> Result<Self> {
        for (i, m) in self.contents.iter_mut().enumerate() {
            m.filter_map_pitch(|v| f((i, v)))?;
        }
        Ok(self)
    }
}

macro_rules! impl_fns_for_seq {
    ($ty:ident, for $($fn:ident)*) => ($(
        fn $fn(mut self, n: $ty) -> Self {
            self.contents.iter_mut().for_each(|p| {
                p.$fn(n);
            });
            self
        }
    )*)
}

impl<T> Pitch<T> for Melody<T>
where
    T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum,
{
    impl_fns_for_seq!(T, for transpose_pitch invert_pitch modulus trim_min trim_max bounce_min bounce_max);

    fn set_pitches(mut self, p: Vec<T>) -> Result<Self> {
        for m in self.contents.iter_mut() {
            m.set_pitches(p.clone())?;
        }
        Ok(self)
    }

    fn map_pitch<MapT: Fn(&T) -> T>(mut self, f: MapT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            m.map_pitch(&f);
        });
        self
    }

    fn filter_pitch<FilterT: Fn(&T) -> bool>(mut self, f: FilterT) -> Result<Self> {
        for m in self.contents.iter_mut() {
            m.filter_pitch(&f)?;
        }
        Ok(self)
    }

    fn filter_map_pitch<MapT: Fn(&T) -> Option<T>>(mut self, f: MapT) -> Result<Self> {
        for m in self.contents.iter_mut() {
            m.filter_map_pitch(&f)?;
        }
        Ok(self)
    }

    fn augment_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            m.augment_pitch(n);
        });
        self
    }

    fn diminish_pitch<AT: AugDim<T> + Copy>(mut self, n: AT) -> Self {
        self.contents.iter_mut().for_each(|m| {
            m.diminish_pitch(n);
        });
        self
    }

    fn trim(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|m| {
            m.trim(first, second);
        });
        self
    }

    fn bounce(mut self, first: T, second: T) -> Self {
        self.contents.iter_mut().for_each(|m| {
            m.bounce(first, second);
        });
        self
    }

    fn is_silent(self) -> bool {
        // TODO make this work better
        self.contents
            .iter()
            .all(|m| m.values.is_empty() || m.volume == 0)
    }
}

impl<T> Melody<T>
where
    T: Pitch<T> + Clone + Copy + Num + Debug + PartialOrd + AddAssign + Bounded + Sum,
{
    pub fn to_volume(&self) -> Vec<u8> {
        self.contents.iter().map(|m| m.volume).collect()
    }

    pub fn to_duration(&self) -> Vec<u32> {
        self.contents.iter().map(|m| m.timing.duration).collect()
    }

    pub fn to_ticks(&self) -> Result<Vec<(u32, u32)>> {
        let mut curr = 0;
        self.contents
            .iter()
            .map(|m| {
                let start = m.timing.start_tick(curr)?;
                let end = m.timing.end_tick(curr)?;

                curr = m.timing.next_tick(curr)?;

                Ok((start, end))
            })
            .collect()
    }

    pub fn to_start_ticks(&self) -> Result<Vec<u32>> {
        let mut curr = 0;
        self.contents
            .iter()
            .map(|m| {
                let start = m.timing.start_tick(curr)?;

                curr = m.timing.next_tick(curr)?;

                Ok(start)
            })
            .collect()
    }

    pub fn to_end_ticks(&self) -> Result<Vec<u32>> {
        let mut curr = 0;
        self.contents
            .iter()
            .map(|m| {
                let end = m.timing.end_tick(curr)?;

                curr = m.timing.next_tick(curr)?;

                Ok(end)
            })
            .collect()
    }

    pub fn last_tick(&self) -> Result<u32> {
        let mut last = self.metadata.last_tick(0)?;
        let mut curr = 0;

        for m in self.contents.iter() {
            let thislast = m.last_tick(curr)?;
            if thislast > last {
                last = thislast;
            }
            curr = m.timing.next_tick(curr)?;
        }

        Ok(last)
    }

    pub fn max_volume(&self) -> Option<u8> {
        self.contents
            .iter()
            .filter(|m| !m.values.is_empty())
            .max_by(|a, b| a.volume.cmp(&b.volume))
            .map(|r| r.volume)
    }

    pub fn min_volume(&self) -> Option<u8> {
        self.contents
            .iter()
            .filter(|m| !m.values.is_empty())
            .min_by(|a, b| a.volume.cmp(&b.volume))
            .map(|r| r.volume)
    }

    pub fn with_volume(self, vel: u8) -> Self {
        self.mutate_each(|m| m.volume = vel)
    }

    pub fn with_volumes(mut self, vel: Vec<u8>) -> Result<Self> {
        if vel.len() != self.contents.len() {
            return Err(anyhow!(
                "supplied volumes are of a different length ({:?}) from Sequence ({:?})",
                vel.len(),
                self.contents.len()
            ));
        }

        for (m, v) in self.contents.iter_mut().zip(vel.iter()) {
            m.volume = *v;
        }

        Ok(self)
    }

    pub fn with_volume_at(self, ix: &[i32], vel: u8) -> Result<Self> {
        self.mutate_indices(ix, move |m| m.volume = vel)
    }

    pub fn with_duration(self, dur: u32) -> Self {
        self.mutate_each(|m| m.timing.duration = dur)
    }

    pub fn with_durations(mut self, dur: Vec<u32>) -> Result<Self> {
        if dur.len() != self.contents.len() {
            return Err(anyhow!(
                "supplied volumes are of a different length ({:?}) from Sequence ({:?})",
                dur.len(),
                self.contents.len()
            ));
        }

        for (m, v) in self.contents.iter_mut().zip(dur.iter()) {
            m.timing.duration = *v;
        }

        Ok(self)
    }

    pub fn with_duration_at(self, ix: &[i32], dur: u32) -> Result<Self> {
        self.mutate_indices(ix, move |m| m.timing.duration = dur)
    }

    pub fn with_exact_tick_at(self, ix: &[i32], tick: u32) -> Result<Self> {
        self.mutate_indices(ix, move |m| m.timing.tick = Some(tick))
    }

    pub fn with_start_tick(self, tick: u32) -> Result<Self> {
        self.with_exact_tick_at(&[0], tick)
    }

    pub fn with_event_at(self, ix: &[i32], evt: Metadata) -> Result<Self> {
        self.mutate_indices(ix, |m| {
            m.with_event(&evt.clone());
        })
    }

    pub fn augment_rhythm(self, a: u32) -> Result<Self, String> {
        let fi32 = algorithms::augment(&a);
        let fu32 = algorithms::augment(&a);

        Ok(self.mutate_each(|m| {
            m.mutate_exact_tick(&fu32)
                .mutate_duration(&fu32)
                .mutate_offset(&fi32);
        }))
    }

    pub fn diminish_rhythm(self, a: u32) -> Result<Self, String> {
        let fi32 = algorithms::diminish(&a)?;
        let fu32 = algorithms::diminish(&a)?;

        Ok(self.mutate_each(|m| {
            m.mutate_exact_tick(&fu32)
                .mutate_duration(&fu32)
                .mutate_offset(&fi32);
        }))
    }

    // Join consecutive chords together if the passed function is true for
    // (chord a, chord a + 1); if chords are joined non-pitch information
    // about the second chord will be lost.
    pub fn join_if(self, f: fn(&MelodyMember<T>, &MelodyMember<T>) -> bool) -> Self {
        self.mutate_contents(|c| {
            for i in (0..c.len() - 1).rev() {
                if f(&c[i], &c[i + 1]) {
                    c[i].timing.duration += c[i + 1].timing.duration;
                    c.remove(i + 1);
                }
            }
        })
    }

    // Join consecutive chords together if they are identical; if chords are
    // joined non-pitch information about the second chord will be lost.
    pub fn join_repeats(self) -> Self {
        self.join_if(|a, b| a.values == b.values)
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::entities::timing::DurationalEventTiming;
    use crate::metadata::{Metadata, MetadataData, MetadataList};
    use crate::ops::pitch::Pitch;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    use super::*;

    #[test]
    fn mm_macros() {
        assert_eq!(melody_member!(), MelodyMember::<i32>::default());
        assert_eq!(melody_member!(None), MelodyMember::<i32>::default());
        assert_eq!(melody_member!(vec![]), MelodyMember::<i32>::default());
        assert_eq!(
            melody_member!(5),
            MelodyMember {
                values: vec![5],
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!(Some(5)),
            MelodyMember {
                values: vec![5],
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!(vec!(5, 6, 7)),
            MelodyMember {
                values: vec![5, 6, 7],
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!([[5, 6, 7]]),
            MelodyMember {
                values: vec![5, 6, 7],
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!(5, 96),
            MelodyMember {
                values: vec![5],
                volume: 96,
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!([[5, 6, 7]], 96),
            MelodyMember {
                values: vec![5, 6, 7],
                volume: 96,
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!(5, 96, 256),
            MelodyMember {
                values: vec![5],
                volume: 96,
                timing: DurationalEventTiming::new(256, None, 0),
                ..Default::default()
            }
        );
        assert_eq!(
            melody_member!([[5, 6, 7]], 96, 256),
            MelodyMember {
                values: vec![5, 6, 7],
                volume: 96,
                timing: DurationalEventTiming::new(256, None, 0),
                ..Default::default()
            }
        );
    }

    #[test]
    fn mm_from() {
        assert_eq!(MelodyMember::from(5).values, vec![5]);
        assert_eq!(MelodyMember::from(Some(5)).values, vec![5]);
        assert_eq!(MelodyMember::<i32>::from(None).values, vec![]);
        assert_eq!(MelodyMember::from(vec![5]).values, vec![5]);
    }

    #[test]
    fn mm_with_event() {
        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_event(&Metadata::try_from(("text", "test text")).unwrap()),
            &MelodyMember {
                values: vec![12, 16],
                timing: DurationalEventTiming::default(),
                volume: DEFAULT_VOLUME,
                before: MetadataList::new(vec![Metadata::try_from(("text", "test text")).unwrap()])
            }
        );
    }

    #[test]
    fn mm_last_tick() {
        assert_eq!(
            melody_member!([[12, 16]], 32, 64).last_tick(96).unwrap(),
            160,
            "duration only"
        );

        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_duration(64)
                .with_exact_tick(80)
                .last_tick(0)
                .unwrap(),
            144,
            "duration and exact tick"
        );

        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_duration(64)
                .with_exact_tick(80)
                .last_tick(96)
                .unwrap(),
            144,
            "duration and exact tick"
        );

        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_duration(64)
                .with_event(&Metadata::try_from(("tempo", 120)).unwrap())
                .last_tick(0)
                .unwrap(),
            64,
            "duration, untimed event"
        );

        assert_eq!(
            melody_member!([[12, 16]], 32, 64)
                .with_duration(64)
                .with_event(
                    Metadata::try_from(("tempo", 120))
                        .unwrap()
                        .with_exact_tick(80)
                )
                .last_tick(0)
                .unwrap(),
            80,
            "duration, event with exact tick, event after note ends"
        );

        assert_eq!(
            melody_member!([[12, 16]], 32, 64)
                .with_event(
                    Metadata::try_from(("tempo", 120))
                        .unwrap()
                        .with_exact_tick(80)
                )
                .last_tick(120)
                .unwrap(),
            184,
            "duration, event with exact tick, event before note ends"
        );

        assert_eq!(
            melody_member!([[12, 16]], 32, 64)
                .with_event(Metadata::try_from(("tempo", 120)).unwrap().with_offset(80))
                .with_event(
                    Metadata::try_from(("tempo", 120))
                        .unwrap()
                        .with_exact_tick(160)
                )
                .last_tick(0)
                .unwrap(),
            160,
            "duration, events with exact tick and offset, exact tick event last"
        );

        assert_eq!(
            melody_member!([[12, 16]], 32, 128)
                .with_duration(128)
                .with_event(Metadata::try_from(("tempo", 120)).unwrap().with_offset(80))
                .with_event(
                    Metadata::try_from(("tempo", 120))
                        .unwrap()
                        .with_exact_tick(160)
                )
                .last_tick(100)
                .unwrap(),
            228,
            "duration, events with exact tick and offset, note ends last"
        );

        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_duration(64)
                .with_offset(16)
                .with_exact_tick(80)
                .last_tick(96)
                .unwrap(),
            160,
            "duration, offset and last tick"
        );

        assert_eq!(
            MelodyMember::<i32>::from(vec![])
                .with_duration(64)
                .with_offset(16)
                .with_exact_tick(80)
                .last_tick(96)
                .unwrap(),
            0,
            "no notes, no metadata events, duration, offset and last tick"
        );

        assert_eq!(
            MelodyMember::<i32>::from(vec![])
                .with_duration(64)
                .with_event(Metadata::try_from(("tempo", 120)).unwrap().with_offset(80))
                .last_tick(96)
                .unwrap(),
            176,
            "no notes, metadata event, duration, offset and last tick"
        );
    }

    #[test]
    fn mm_set_pitches() {
        assert_eq!(
            MelodyMember::from(vec![])
                .set_pitches(vec![5, 12, 16])
                .unwrap(),
            &MelodyMember::from(vec![5, 12, 16])
        );
    }

    #[test]
    fn mm_map_pitch() {
        assert_eq!(
            MelodyMember::from(vec![5, 12, 16]).map_pitch(|p| p + 5),
            &MelodyMember::from(vec![10, 17, 21]),
        );
    }

    #[test]
    fn mm_filter_pitch() {
        assert_eq!(
            MelodyMember::from(vec![5, 12, 16])
                .filter_pitch(|p| *p > 10)
                .unwrap(),
            &MelodyMember::from(vec![12, 16]),
        );
    }

    #[test]
    fn mm_augment_pitch() {
        assert_eq!(
            MelodyMember::from(vec![5, 12, 16]).augment_pitch(2),
            &MelodyMember::from(vec![10, 24, 32]),
        );

        assert_eq!(
            MelodyMember::from(vec![5, 12, 16]).augment_pitch(2.5),
            &MelodyMember::from(vec![12, 30, 40]),
        );

        assert_eq!(
            MelodyMember::from(vec![5.5, 12.0, 16.5]).augment_pitch(2),
            &MelodyMember::from(vec![11.0, 24.0, 33.0]),
        );

        assert_eq!(
            MelodyMember::from(vec![5.5, 12.0, 16.5]).augment_pitch(2.5),
            &MelodyMember::from(vec![13.75, 30.0, 41.25]),
        );
    }

    #[test]
    fn mm_diminish_pitch() {
        assert_eq!(
            MelodyMember::from(vec![10, 24, 32]).diminish_pitch(2),
            &MelodyMember::from(vec![5, 12, 16]),
        );

        assert_eq!(
            MelodyMember::from(vec![12, 30, 40]).diminish_pitch(2.5),
            &MelodyMember::from(vec![4, 12, 16]),
        );

        assert_eq!(
            MelodyMember::from(vec![11.0, 24.0, 33.0]).diminish_pitch(2),
            &MelodyMember::from(vec![5.5, 12.0, 16.5]),
        );

        assert_eq!(
            MelodyMember::from(vec![13.75, 30.0, 41.25]).diminish_pitch(2.5),
            &MelodyMember::from(vec![5.5, 12.0, 16.5]),
        );
    }

    #[test]
    fn mm_trim_pitch() {
        assert_eq!(
            MelodyMember::from(vec![10, 24, 32]).trim(12, 30),
            &MelodyMember::from(vec![12, 24, 30]),
        );
    }

    #[test]
    fn mm_bounce_pitch() {
        assert_eq!(
            MelodyMember::from(vec![2, 10, 24, 32, 42]).bounce(20, 30),
            &MelodyMember::from(vec![22, 30, 24, 28, 22]),
        );
    }

    #[test]
    fn is_silent() {
        assert!(!MelodyMember::from(vec![10, 24, 32]).is_silent());
        assert!(MelodyMember::<i32>::from(vec![]).is_silent());
    }

    #[test]
    fn from_macro() {
        assert_eq!(
            melody![5, 12, 16],
            Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ])
        );

        assert_eq!(
            melody![[5], [], [12, 16]],
            Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ])
        );
    }

    #[test]
    fn try_from_vec() {
        assert_eq!(
            Melody::try_from(vec![5, 12, 16]).unwrap(),
            melody![5, 12, 16]
        );
    }

    #[test]
    fn try_from_vec_of_options() {
        assert_eq!(
            Melody::try_from(vec![Some(5), None, Some(12), Some(16)]).unwrap(),
            melody![[5], [], [12], [16]]
        );
    }

    #[test]
    fn try_from_vec_of_vecs() {
        assert_eq!(
            Melody::try_from(vec![vec![5], vec![], vec![12, 16]]).unwrap(),
            melody![[5], [], [12, 16]]
        );
    }

    #[test]
    fn try_from_vec_of_members() {
        assert_eq!(
            Melody::<i32>::try_from(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ])
            .unwrap(),
            melody![[5], [], [12, 16]]
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert_eq!(
            Melody::try_from(ChordSeq {
                contents: vec![vec![5], vec![], vec![12, 16]],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            Melody {
                contents: vec![
                    MelodyMember::from(vec![5]),
                    MelodyMember::from(vec![]),
                    MelodyMember::from(vec![12, 16]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            Melody::try_from(NumericSeq {
                contents: vec![5, 12, 16],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            Melody {
                contents: vec![
                    MelodyMember::from(vec![5]),
                    MelodyMember::from(vec![12]),
                    MelodyMember::from(vec![16]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert_eq!(
            Melody::try_from(NoteSeq {
                contents: vec![Some(5), None, Some(12), Some(16)],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            })
            .unwrap(),
            Melody {
                contents: vec![
                    MelodyMember::from(vec![5]),
                    MelodyMember::from(vec![]),
                    MelodyMember::from(vec![12]),
                    MelodyMember::from(vec![16]),
                ],
                metadata: MetadataList::new(vec![MetadataData::Tempo(144.0).into()])
            }
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            melody![[5], [], [12, 16]].to_flat_pitches(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            melody![[5], [], [12, 16]].to_pitches(),
            vec![vec![5], vec![], vec![12, 16]]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert!(Melody::<i32>::new(vec![MelodyMember::from(vec![])])
            .to_numeric_values()
            .is_err());
        assert!(melody![[12, 16]].to_numeric_values().is_err());
        assert_eq!(
            melody![5, 12, 16].to_numeric_values().unwrap(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert!(melody![[12, 16]].to_optional_numeric_values().is_err());
        assert_eq!(
            melody![[5], [], [12], [16]]
                .to_optional_numeric_values()
                .unwrap(),
            vec![Some(5), None, Some(12), Some(16)]
        );
    }

    #[test]
    fn to_volume() {
        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: MetadataList::new(vec![]),
                },
            ])
            .to_volume(),
            vec![25, 35]
        );
    }

    #[test]
    fn to_duration() {
        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(16, None, 0),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(32, None, 0),
                    volume: 35,
                    before: MetadataList::new(vec![]),
                },
            ])
            .to_duration(),
            vec![16, 32]
        );
    }

    macro_rules! mmtiming {
        ($tick:expr, $offset:expr, $duration:expr) => {
            MelodyMember {
                values: vec![20],
                timing: DurationalEventTiming::new($duration, $tick, $offset),
                volume: 32,
                before: MetadataList::new(vec![]),
            }
        };
    }

    #[test]
    fn to_ticks() {
        assert_eq!(
            Melody::new(vec![
                mmtiming!(None, 0, 64),
                mmtiming!(None, 16, 32),
                mmtiming!(None, -48, 96),
                mmtiming!(Some(128), 0, 32),
                mmtiming!(None, 0, 64),
                mmtiming!(Some(160), 32, 96),
                mmtiming!(None, 0, 32),
                mmtiming!(None, -64, 32),
            ])
            .to_ticks()
            .unwrap(),
            vec![
                (0, 64),
                (80, 112),
                (48, 144),
                (128, 160),
                (160, 224),
                (192, 288),
                (256, 288),
                (224, 256)
            ]
        );
    }

    #[test]
    fn to_start_ticks() {
        assert_eq!(
            Melody::new(vec![
                mmtiming!(None, 0, 64),
                mmtiming!(None, 16, 32),
                mmtiming!(None, -48, 96),
                mmtiming!(Some(128), 0, 32),
                mmtiming!(None, 0, 64),
                mmtiming!(Some(160), 32, 96),
                mmtiming!(None, 0, 32),
                mmtiming!(None, -64, 32),
            ])
            .to_start_ticks()
            .unwrap(),
            vec![0, 80, 48, 128, 160, 192, 256, 224,]
        );
    }

    #[test]
    fn to_end_ticks() {
        assert_eq!(
            Melody::new(vec![
                mmtiming!(None, 0, 64),
                mmtiming!(None, 16, 32),
                mmtiming!(None, -48, 96),
                mmtiming!(Some(128), 0, 32),
                mmtiming!(None, 0, 64),
                mmtiming!(Some(160), 32, 96),
                mmtiming!(None, 0, 32),
                mmtiming!(None, -64, 32),
            ])
            .to_end_ticks()
            .unwrap(),
            vec![64, 112, 144, 160, 224, 288, 288, 256]
        );
    }

    #[test]
    fn max_volume() {
        assert!(Melody::<i32>::new(vec![]).max_volume().is_none());
        assert!(Melody::<i32>::new(vec![MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            volume: 25,
            before: MetadataList::new(vec![]),
        }])
        .max_volume()
        .is_none());

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![],
                    timing: DurationalEventTiming::default(),
                    volume: 45,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: MetadataList::new(vec![]),
                },
            ])
            .max_volume(),
            Some(35)
        );
    }

    #[test]
    fn min_volume() {
        assert!(Melody::<i32>::new(vec![]).min_volume().is_none());
        assert!(Melody::<i32>::new(vec![MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            volume: 25,
            before: MetadataList::new(vec![]),
        }])
        .min_volume()
        .is_none());

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![],
                    timing: DurationalEventTiming::default(),
                    volume: 15,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: MetadataList::new(vec![]),
                },
            ])
            .min_volume(),
            Some(25)
        );
    }

    #[test]
    fn with_volume() {
        assert_eq!(
            Melody::try_from(vec![12, 16]).unwrap().with_volume(25),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_volumes() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_volumes(vec![25, 35])
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_volume_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_volume_at(&[-1], 25)
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_duration() {
        assert_eq!(
            Melody::try_from(vec![12, 16]).unwrap().with_duration(25),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default().with_duration(25),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(25, None, 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_durations() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_durations(vec![25, 35])
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(25, None, 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(35, None, 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_duration_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_duration_at(&[-1], 25)
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(25, None, 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_exact_tick_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_exact_tick_at(&[-1], 25)
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(0, Some(25), 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_start_tick() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_start_tick(25)
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(0, Some(25), 0),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_event_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_event_at(&[-1], Metadata::try_from(("key-signature", "D")).unwrap())
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![
                        Metadata::try_from(("key-signature", "D")).unwrap()
                    ]),
                },
            ])
        );
    }

    #[test]
    fn augment_rhythm() {
        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(32, None, 100),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        16
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(25, None, 75),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(32),
                        25
                    )]),
                },
            ])
            .augment_rhythm(3)
            .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(96, None, 300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(75, None, 225),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(96),
                        75
                    )]),
                }
            ])
        );
    }

    #[test]
    fn diminish_rhythm() {
        assert!(Melody::try_from(Vec::<i32>::new())
            .unwrap()
            .diminish_rhythm(0)
            .is_err());

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(96, None, 300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(75, None, 225),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(96),
                        75
                    )]),
                }
            ])
            .diminish_rhythm(3)
            .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(32, None, 100),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        16
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(25, None, 75),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(32),
                        25
                    )]),
                },
            ])
        );
    }

    #[test]
    fn join_if() {
        assert_eq!(
            Melody::new(vec![
                melody_member!(12, 20, 32),
                melody_member!(12, 30, 64),
                melody_member!(16, 40, 32),
                melody_member!(12, 50, 32),
                melody_member!(12, 60, 64),
                melody_member!(12, 70, 128)
            ])
            .join_if(|a, b| a.values == b.values),
            Melody::new(vec![
                melody_member!(12, 20, 96),
                melody_member!(16, 40, 32),
                melody_member!(12, 50, 224)
            ])
        );
    }

    #[test]
    fn join_repeats() {
        assert_eq!(
            Melody::new(vec![
                melody_member!(12, 20, 32),
                melody_member!(12, 30, 64),
                melody_member!(16, 40, 32),
                melody_member!(12, 50, 32),
                melody_member!(12, 60, 64),
                melody_member!(12, 70, 128)
            ])
            .join_repeats(),
            Melody::new(vec![
                melody_member!(12, 20, 96),
                melody_member!(16, 40, 32),
                melody_member!(12, 50, 224)
            ])
        );
    }

    #[test]
    fn set_pitches() {
        assert_eq!(
            Melody::<i32>::new(vec![]).set_pitches(vec![55]).unwrap(),
            Melody::<i32>::new(vec![])
        );
        assert_eq!(
            melody![4, 2, 5, 6, 3].set_pitches(vec![]).unwrap(),
            melody![[], [], [], [], []]
        );
        assert_eq!(
            melody![4, 2, 5, 6, 3].set_pitches(vec![55]).unwrap(),
            melody![55, 55, 55, 55, 55]
        );
        assert_eq!(
            melody![4, 2, 5, 6, 3].set_pitches(vec![55, 66]).unwrap(),
            melody![[55, 66], [55, 66], [55, 66], [55, 66], [55, 66]]
        );

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::new(96, None, 300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::new(75, None, 225),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(96),
                        75
                    )]),
                }
            ])
            .set_pitches(vec![55])
            .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![55],
                    timing: DurationalEventTiming::new(96, None, 300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![55],
                    timing: DurationalEventTiming::new(75, None, 225),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(false),
                        Some(96),
                        75
                    )]),
                }
            ])
        );
    }
}
