use crate::algorithms::algorithms;
use crate::collections::event::{EventList, MetaEvent};
use crate::collections::traits::Collection;
use crate::default_methods;
use crate::entities::timing::DurationalEventTiming;
use crate::sequences::chord::ChordSeq;
use crate::sequences::note::NoteSeq;
use crate::sequences::numeric::NumericSeq;
use crate::sequences::traits::Sequence;

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
    before: EventList,
}

impl<T> Default for MelodyMember<T> {
    fn default() -> Self {
        MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            volume: DEFAULT_VOLUME,
            before: EventList::new(vec![]),
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
    fn with_event(mut self, e: MetaEvent) -> Self {
        self.before = self.before.append_items(&[e]);

        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Melody<T> {
    contents: Vec<MelodyMember<T>>,
}

#[derive(Debug, PartialEq)]
pub enum MelodyError {
    InvalidValues,
}

impl<T> TryFrom<Vec<Vec<T>>> for Melody<T>
where
    T: Num + Debug + PartialOrd,
{
    type Error = MelodyError;

    fn try_from(what: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.into_iter().map(|v| MelodyMember::from(v)).collect(),
        })
    }
}

impl<T> TryFrom<Vec<T>> for Melody<T>
where
    T: Num + Debug + PartialOrd,
{
    type Error = MelodyError;

    fn try_from(what: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self {
            contents: what.into_iter().map(|v| MelodyMember::from(v)).collect(),
        })
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
                Ok(Self {
                    contents: what
                        .to_pitches()
                        .into_iter()
                        .map(|v| MelodyMember::from(v))
                        .collect(),
                })
            }
        }
    };
}

try_from_seq!(NumericSeq<T>);
try_from_seq!(NoteSeq<T>);
try_from_seq!(ChordSeq<T>);

impl<T: Clone + Num + Debug + PartialOrd + Bounded> Collection<MelodyMember<T>> for Melody<T> {
    default_methods!(MelodyMember<T>);
}

impl<T: Clone + Copy + Num + Debug + PartialOrd + Bounded + Sum + From<i32>>
    Sequence<MelodyMember<T>, T> for Melody<T>
{
    fn mutate_pitches<F: Fn(&mut T)>(self, f: F) -> Self {
        self.mutate_each(|m| {
            for p in m.values.iter_mut() {
                f(p);
            }
        })
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

    pub fn to_duration(&self) -> Vec<u32> {
        self.contents.iter().map(|m| m.timing.duration).collect()
    }

    pub fn with_volume(self, vel: u8) -> Self {
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

    pub fn with_duration(self, dur: u32) -> Self {
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

    pub fn with_event_at(self, ix: &[i32], evt: MetaEvent) -> Result<Self, String> {
        self.mutate_indices(ix, |m| {
            *m = m.clone().with_event(evt.clone());
        })
    }

    pub fn augment_rhythm(self, a: u32) -> Result<Self, String> {
        let fi32 = algorithms::augment(&a);
        let fu32 = algorithms::augment(&a);

        Ok(self.mutate_each(|m| {
            fi32(&mut m.timing.offset);
            fu32(&mut m.timing.duration);
        }))
    }

    pub fn diminish_rhythm(self, a: u32) -> Result<Self, String> {
        let fi32 = algorithms::diminish(&a)?;
        let fu32 = algorithms::diminish(&a)?;

        Ok(self.mutate_each(|m| {
            fi32(&mut m.timing.offset);
            fu32(&mut m.timing.duration);
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::event::{EventList, MetaEvent};
    use crate::collections::traits::Collection;
    use crate::entities::timing::{DurationalEventTiming, Timing};
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember, DEFAULT_VOLUME};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

    #[test]
    fn mel_member_with_event() {
        assert_eq!(
            MelodyMember::from(vec![12, 16])
                .with_event(MetaEvent::try_from(("text", "test text")).unwrap()),
            MelodyMember {
                values: vec![12, 16],
                timing: DurationalEventTiming::default(),
                volume: DEFAULT_VOLUME,
                before: EventList::new(vec![MetaEvent::try_from(("text", "test text")).unwrap()])
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: EventList::new(vec![]),
                },
            ])
            .to_volume(),
            vec![25, 35]
        );
    }

    #[test]
    fn max_volume() {
        assert!(Melody::<i32>::new(vec![]).max_volume().is_none());
        assert!(Melody::<i32>::new(vec![MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            volume: 25,
            before: EventList::new(vec![]),
        }])
        .max_volume()
        .is_none());

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![],
                    timing: DurationalEventTiming::default(),
                    volume: 45,
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: EventList::new(vec![]),
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
            before: EventList::new(vec![]),
        }])
        .min_volume()
        .is_none());

        assert_eq!(
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![],
                    timing: DurationalEventTiming::default(),
                    volume: 15,
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: EventList::new(vec![]),
                },
            ])
            .min_volume(),
            Some(25)
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(32),
                    volume: 35,
                    before: EventList::new(vec![]),
                },
            ])
            .to_duration(),
            vec![16, 32]
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: EventList::new(vec![]),
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 35,
                    before: EventList::new(vec![]),
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: 25,
                    before: EventList::new(vec![]),
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(25),
                    volume: DEFAULT_VOLUME,
                    before: EventList::new(vec![]),
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(35),
                    volume: DEFAULT_VOLUME,
                    before: EventList::new(vec![]),
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
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(25),
                    volume: DEFAULT_VOLUME,
                    before: EventList::new(vec![]),
                },
            ])
        );
    }

    #[test]
    fn with_event_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_event_at(&[-1], MetaEvent::try_from(("key-signature", "D")).unwrap())
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: EventList::new(vec![]),
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    volume: DEFAULT_VOLUME,
                    before: EventList::new(vec![
                        MetaEvent::try_from(("key-signature", "D")).unwrap()
                    ]),
                },
            ])
        );
    }

    #[test]
    fn augment_rhythm() {
        assert_eq!(
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![12],
                        timing: DurationalEventTiming::default()
                            .with_duration(32)
                            .with_offset(100),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                    MelodyMember {
                        values: vec![16],
                        timing: DurationalEventTiming::default()
                            .with_duration(25)
                            .with_offset(75),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    }
                ]
            }
            .augment_rhythm(3)
            .unwrap(),
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![12],
                        timing: DurationalEventTiming::default()
                            .with_duration(96)
                            .with_offset(300),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                    MelodyMember {
                        values: vec![16],
                        timing: DurationalEventTiming::default()
                            .with_duration(75)
                            .with_offset(225),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                ]
            }
        );
    }

    #[test]
    fn diminish_rhythm() {
        assert!(Melody::try_from(Vec::<i32>::new())
            .unwrap()
            .diminish_rhythm(0)
            .is_err());

        assert_eq!(
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![12],
                        timing: DurationalEventTiming::default()
                            .with_duration(96)
                            .with_offset(300),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                    MelodyMember {
                        values: vec![16],
                        timing: DurationalEventTiming::default()
                            .with_duration(75)
                            .with_offset(225),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    }
                ]
            }
            .diminish_rhythm(3)
            .unwrap(),
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![12],
                        timing: DurationalEventTiming::default()
                            .with_duration(32)
                            .with_offset(100),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                    MelodyMember {
                        values: vec![16],
                        timing: DurationalEventTiming::default()
                            .with_duration(25)
                            .with_offset(75),
                        volume: DEFAULT_VOLUME,
                        before: EventList::new(vec![]),
                    },
                ]
            }
        );
    }
}
