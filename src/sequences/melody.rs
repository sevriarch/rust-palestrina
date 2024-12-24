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

pub const DEFAULT_VELOCITY: u8 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct MelodyMember<T> {
    values: Vec<T>,
    timing: DurationalEventTiming,
    velocity: u8,
}

impl<T> Default for MelodyMember<T> {
    fn default() -> Self {
        MelodyMember {
            values: vec![],
            timing: DurationalEventTiming::default(),
            velocity: DEFAULT_VELOCITY,
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
    fn mutate_pitches<F: Fn(&mut T)>(mut self, f: F) -> Self {
        for v in self.contents.iter_mut() {
            for p in v.values.iter_mut() {
                f(p);
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
    pub fn with_velocity(mut self, vel: u8) -> Self {
        for m in self.contents.iter_mut() {
            m.velocity = vel;
        }
        self
    }

    pub fn with_velocities(mut self, vel: Vec<u8>) -> Result<Self, String> {
        if vel.len() != self.contents.len() {
            return Err(format!(
                "supplied velocities are of a different length ({:?}) from Sequence ({:?})",
                vel.len(),
                self.contents.len()
            ));
        }

        for (m, v) in self.contents.iter_mut().zip(vel.iter()) {
            m.velocity = *v;
        }

        Ok(self)
    }

    pub fn with_velocity_at(self, ix: &[i32], vel: u8) -> Result<Self, String> {
        self.mutate_indices(ix, move |m| m.velocity = vel)
    }

    pub fn with_duration(mut self, dur: u32) -> Self {
        for m in self.contents.iter_mut() {
            m.timing.duration = dur;
        }
        self
    }

    pub fn with_durations(mut self, dur: Vec<u32>) -> Result<Self, String> {
        if dur.len() != self.contents.len() {
            return Err(format!(
                "supplied velocities are of a different length ({:?}) from Sequence ({:?})",
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
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;
    use crate::entities::timing::DurationalEventTiming;
    use crate::sequences::chord::ChordSeq;
    use crate::sequences::melody::{Melody, MelodyMember, DEFAULT_VELOCITY};
    use crate::sequences::note::NoteSeq;
    use crate::sequences::numeric::NumericSeq;
    use crate::sequences::traits::Sequence;

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
    fn with_velocity() {
        assert_eq!(
            Melody::try_from(vec![12, 16]).unwrap().with_velocity(25),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    velocity: 25,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    velocity: 25,
                },
            ])
        );
    }

    #[test]
    fn with_velocities() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_velocities(vec![25, 35])
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    velocity: 25,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    velocity: 35,
                },
            ])
        );
    }

    #[test]
    fn with_velocity_at() {
        assert_eq!(
            Melody::try_from(vec![12, 16])
                .unwrap()
                .with_velocity_at(&[-1], 25)
                .unwrap(),
            Melody::new(vec![
                MelodyMember {
                    values: vec![12],
                    timing: DurationalEventTiming::default(),
                    velocity: DEFAULT_VELOCITY,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default(),
                    velocity: 25,
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
                    velocity: DEFAULT_VELOCITY,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(25),
                    velocity: DEFAULT_VELOCITY,
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
                    velocity: DEFAULT_VELOCITY,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(35),
                    velocity: DEFAULT_VELOCITY,
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
                    velocity: DEFAULT_VELOCITY,
                },
                MelodyMember {
                    values: vec![16],
                    timing: DurationalEventTiming::default().with_duration(25),
                    velocity: DEFAULT_VELOCITY,
                },
            ])
        );
    }
}
