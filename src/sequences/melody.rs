use crate::algorithms;
use crate::collections::traits::Collection;
use crate::entities::timing::{DurationalEventTiming, Timing};
use crate::metadata::{data::Metadata, list::MetadataList};
use crate::sequences::chord::ChordSeq;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;
use crate::{default_collection_methods, default_sequence_methods};

use num_traits::{Bounded, Num};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::iter::Sum;

pub const DEFAULT_VOLUME: u8 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct MelodyMember<T> {
    values: Vec<T>,
    timing: DurationalEventTiming,
    volume: u8,
    before: MetadataList,
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

impl<T> From<T> for MelodyMember<T> {
    fn from(what: T) -> Self {
        MelodyMember {
            values: vec![what],
            ..Default::default()
        }
    }
}

impl<T> From<Vec<T>> for MelodyMember<T> {
    fn from(what: Vec<T>) -> Self {
        MelodyMember {
            values: what,
            ..Default::default()
        }
    }
}

impl<T> MelodyMember<T> {
    fn with_event(mut self, e: Metadata) -> Self {
        self.before = self.before.append(e);

        self
    }

    fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> &mut Self {
        for m in self.before.contents.iter_mut() {
            m.mutate_exact_tick(&f);
        }

        self.timing.mutate_exact_tick(f);
        self
    }

    fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> &mut Self {
        for m in self.before.contents.iter_mut() {
            m.mutate_offset(&f);
        }

        self.timing.mutate_offset(&f);
        self
    }

    fn mutate_duration(&mut self, f: impl Fn(&mut u32)) -> &mut Self {
        self.timing.mutate_duration(f);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Melody<T> {
    contents: Vec<MelodyMember<T>>,
    metadata: MetadataList,
}

#[derive(Debug, PartialEq)]
pub enum MelodyError {
    InvalidValues,
}

impl<T> TryFrom<Vec<Vec<T>>> for Melody<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = MelodyError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Ok(Self::new(
            what.into_iter().map(MelodyMember::from).collect(),
        ))
    }
}

impl<T> TryFrom<Vec<T>> for Melody<T>
where
    T: Clone + Copy + Num + Debug + PartialOrd + Bounded,
{
    type Error = MelodyError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self::new(
            what.into_iter().map(MelodyMember::from).collect(),
        ))
    }
}

macro_rules! try_from_seq {
    ($type:ty) => {
        impl<T> TryFrom<$type> for Melody<T>
        where
            T: Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>,
        {
            type Error = MelodyError;

            fn try_from(what: $type) -> Result<Self, Self::Error> {
                Ok(Self::new(
                    what.to_pitches()
                        .into_iter()
                        .map(MelodyMember::from)
                        .collect(),
                ))
            }
        }
    };
}

try_from_seq!(NumericSeq<T>);
try_from_seq!(NoteSeq<T>);
try_from_seq!(ChordSeq<T>);

impl<T: Clone + Num + Debug + PartialOrd + Bounded> Collection<MelodyMember<T>> for Melody<T> {
    default_collection_methods!(MelodyMember<T>);
    default_sequence_methods!(MelodyMember<T>);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>>
    Sequence<MelodyMember<T>, T> for Melody<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        //self.mutate_each(|m| {
        for c in self.contents.iter_mut() {
            for p in c.values.iter_mut() {
                f(p)
            }
        }
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

    fn to_numeric_values(&self) -> Result<Vec<T>, String> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                1 => Ok(v.values[0]),
                _ => Err("must contain only one value".to_string()),
            })
            .collect()
    }

    fn to_optional_numeric_values(&self) -> Result<Vec<Option<T>>, String> {
        self.contents
            .iter()
            .map(|v| match v.values.len() {
                0 => Ok(None),
                1 => Ok(Some(v.values[0])),
                _ => Err("must contain zero or one values".to_string()),
            })
            .collect()
    }
}

impl<T> Melody<T>
where
    T: Bounded + Clone + Num + Debug + PartialOrd,
{
    pub fn to_volume(&self) -> Vec<u8> {
        self.contents.iter().map(|m| m.volume).collect()
    }

    pub fn to_duration(&self) -> Vec<u32> {
        self.contents.iter().map(|m| m.timing.duration).collect()
    }

    pub fn to_ticks(&self) -> Result<Vec<(u32, u32)>, String> {
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

    pub fn to_start_ticks(&self) -> Result<Vec<u32>, String> {
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

    pub fn to_end_ticks(&self) -> Result<Vec<u32>, String> {
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

    pub fn with_volume(&mut self, vel: u8) -> &Self {
        self.mutate_each(|m| m.volume = vel)
    }

    pub fn with_volumes(mut self, vel: Vec<u8>) -> Result<Self, String> {
        if vel.len() != self.contents.len() {
            return Err(format!(
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

    pub fn with_volume_at(self, ix: &[i32], vel: u8) -> Result<Self, String> {
        self.mutate_indices(ix, move |m| m.volume = vel)
    }

    pub fn with_duration(&mut self, dur: u32) -> &Self {
        self.mutate_each(|m| m.timing.duration = dur)
    }

    pub fn with_durations(mut self, dur: Vec<u32>) -> Result<Self, String> {
        if dur.len() != self.contents.len() {
            return Err(format!(
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

    pub fn with_duration_at(self, ix: &[i32], dur: u32) -> Result<Self, String> {
        self.mutate_indices(ix, move |m| m.timing.duration = dur)
    }

    pub fn with_event_at(self, ix: &[i32], evt: Metadata) -> Result<Self, String> {
        self.mutate_indices(ix, |m| {
            *m = m.clone().with_event(evt.clone());
        })
    }

    pub fn augment_rhythm(&mut self, a: u32) -> Result<&Self, String> {
        let fi32 = algorithms::augment(&a);
        let fu32 = algorithms::augment(&a);

        Ok(self.mutate_each(|m| {
            m.mutate_exact_tick(&fu32)
                .mutate_duration(&fu32)
                .mutate_offset(&fi32);
        }))
    }

    pub fn diminish_rhythm(&mut self, a: u32) -> Result<&Self, String> {
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
    use crate::entities::timing::{DurationalEventTiming, Timing};
    use crate::metadata::{
        data::{Metadata, MetadataData},
        list::MetadataList,
    };
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember, DEFAULT_VOLUME};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn mel_member_with_event() {
        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_event(Metadata::try_from(("text", "test text")).unwrap()),
            MelodyMember {
                values: vec![12, 16],
                timing: DurationalEventTiming::default(),
                volume: DEFAULT_VOLUME,
                before: MetadataList::new(vec![Metadata::try_from(("text", "test text")).unwrap()])
            }
        );
    }

    #[test]
    fn try_from_vec() {
        assert_eq!(
            Melody::try_from(vec![5, 12, 16]),
            Ok(Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ]))
        );
    }

    #[test]
    fn try_from_vec_vec() {
        assert_eq!(
            Melody::try_from(vec![vec![5], vec![], vec![12, 16]]),
            Ok(Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ]))
        );
    }

    #[test]
    fn try_from_chordseq() {
        assert_eq!(
            Melody::try_from(ChordSeq::new(vec![vec![5], vec![], vec![12, 16]])),
            Ok(Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ]))
        );
    }

    #[test]
    fn try_from_numseq() {
        assert_eq!(
            Melody::try_from(NumericSeq::new(vec![5, 12, 16])),
            Ok(Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ]))
        );
    }

    #[test]
    fn try_from_noteseq() {
        assert_eq!(
            Melody::try_from(NoteSeq::new(vec![Some(5), None, Some(12), Some(16)])),
            Ok(Melody::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ]))
        );
    }

    #[test]
    fn to_flat_pitches() {
        assert_eq!(
            Melody::<i32>::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ])
            .to_flat_pitches(),
            vec![5, 12, 16]
        );
    }

    #[test]
    fn to_pitches() {
        assert_eq!(
            Melody::<i32>::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12, 16]),
            ])
            .to_pitches(),
            vec![vec![5], vec![], vec![12, 16]]
        );
    }

    #[test]
    fn to_numeric_values() {
        assert!(Melody::<i32>::new(vec![MelodyMember::from(vec![])])
            .to_numeric_values()
            .is_err());
        assert!(Melody::<i32>::new(vec![MelodyMember::from(vec![12, 16])])
            .to_numeric_values()
            .is_err());
        assert_eq!(
            Melody::<i32>::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ])
            .to_numeric_values(),
            Ok(vec![5, 12, 16])
        );
    }

    #[test]
    fn to_optional_numeric_values() {
        assert!(Melody::<i32>::new(vec![MelodyMember::from(vec![12, 16])])
            .to_optional_numeric_values()
            .is_err());
        assert_eq!(
            Melody::<i32>::new(vec![
                MelodyMember::from(vec![5]),
                MelodyMember::from(vec![]),
                MelodyMember::from(vec![12]),
                MelodyMember::from(vec![16]),
            ])
            .to_optional_numeric_values(),
            Ok(vec![Some(5), None, Some(12), Some(16)])
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
                    timing: DurationalEventTiming::default().with_duration(16),
                    volume: 25,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(32),
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
                timing: DurationalEventTiming {
                    tick: $tick,
                    offset: $offset,
                    duration: $duration,
                },
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
            &Melody::new(vec![
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
            &Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default().with_duration(25),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(25),
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
                    timing: DurationalEventTiming::default().with_duration(25),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(35),
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
                    timing: DurationalEventTiming::default().with_duration(25),
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
                    timing: DurationalEventTiming::default()
                        .with_duration(32)
                        .with_offset(100),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        16
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default()
                        .with_duration(25)
                        .with_offset(75),
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
            &Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default()
                        .with_duration(96)
                        .with_offset(300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default()
                        .with_duration(75)
                        .with_offset(225),
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
                    timing: DurationalEventTiming::default()
                        .with_duration(96)
                        .with_offset(300),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        48
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default()
                        .with_duration(75)
                        .with_offset(225),
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
            &Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default()
                        .with_duration(32)
                        .with_offset(100),
                    volume: DEFAULT_VOLUME,
                    before: MetadataList::new(vec![Metadata::new(
                        MetadataData::Sustain(true),
                        None,
                        16
                    )]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default()
                        .with_duration(25)
                        .with_offset(75),
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

    macro_rules! mmvdv {
        ($p:expr, $d:expr, $v:expr) => {
            MelodyMember {
                values: vec![$p],
                timing: DurationalEventTiming::default().with_duration($d),
                volume: $v,
                before: MetadataList::new(vec![]),
            }
        };
    }

    #[test]
    fn join_if() {
        assert_eq!(
            Melody::new(vec![
                mmvdv!(12, 32, 20),
                mmvdv!(12, 64, 30),
                mmvdv!(16, 32, 40),
                mmvdv!(12, 32, 50),
                mmvdv!(12, 64, 60),
                mmvdv!(12, 128, 70)
            ])
            .join_if(|a, b| a.values == b.values),
            Melody::new(vec![
                mmvdv!(12, 96, 20),
                mmvdv!(16, 32, 40),
                mmvdv!(12, 224, 50)
            ])
        );
    }

    #[test]
    fn join_repeats() {
        assert_eq!(
            Melody::new(vec![
                mmvdv!(12, 32, 20),
                mmvdv!(12, 64, 30),
                mmvdv!(16, 32, 40),
                mmvdv!(12, 32, 50),
                mmvdv!(12, 64, 60),
                mmvdv!(12, 128, 70)
            ])
            .join_repeats(),
            Melody::new(vec![
                mmvdv!(12, 96, 20),
                mmvdv!(16, 32, 40),
                mmvdv!(12, 224, 50)
            ])
        );
    }
}
