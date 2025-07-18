use crate::ops::arithmetic::AugDim;
use anyhow::{anyhow, Result};
use num_traits::{Bounded, Num};
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum TimingError {
    #[error("Impermissible negative tick {0} occurred")]
    NegativeTick(i32),
}

pub trait Timing {
    fn with_exact_tick(&mut self, tick: i32) -> Self;
    fn with_offset(&mut self, tick: i32) -> Self;
    fn mutate_exact_tick(&mut self, f: impl Fn(&mut i32)) -> Self;
    fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> Self;
    fn start_tick(&self, curr: i32) -> Result<i32>;
    fn augment_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self;
    fn diminish_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self;
}

#[inline(always)]
fn pos_or_err(tick: i32) -> Result<i32> {
    if tick < 0 {
        Err(anyhow!(TimingError::NegativeTick(tick)))
    } else {
        Ok(tick)
    }
}

macro_rules! timing_traits {
    () => {
        fn with_exact_tick(&mut self, tick: i32) -> Self {
            self.tick = Some(tick);
            *self
        }

        fn with_offset(&mut self, offset: i32) -> Self {
            self.offset = offset;
            *self
        }

        fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> Self {
            f(&mut self.offset);
            *self
        }

        fn mutate_exact_tick(&mut self, f: impl Fn(&mut i32)) -> Self {
            if let Some(tick) = self.tick.as_mut() {
                f(tick);
            }
            *self
        }

        fn start_tick(&self, curr: i32) -> Result<i32> {
            pos_or_err(self.offset + self.tick.unwrap_or(curr))
        }
    };
}

/// A structure with timing information for a single non-durational event.
/// tick is an optional exact tick that the event is anchored to; offset is a
/// timing offset
///
/// Both of these are measured in MIDI ticks.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EventTiming {
    pub tick: Option<i32>,
    pub offset: i32,
}

impl Timing for EventTiming {
    timing_traits!();

    fn augment_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self {
        if let Some(ref mut t) = self.tick {
            v.augment_target(t);
        }

        v.augment_target(&mut self.offset);
        *self
    }

    fn diminish_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self {
        if let Some(ref mut t) = self.tick {
            v.diminish_target(t);
        }

        v.diminish_target(&mut self.offset);
        *self
    }
}

impl EventTiming {
    pub fn new(tick: Option<i32>, offset: i32) -> Self {
        Self { tick, offset }
    }
}

/// A structure with timing information for a single non-durational even.
/// tick is an optional exact tick that the event is anchored to; offset is a
/// timing offset and duration is the length of the event.
///
/// All of these are measured in MIDI ticks.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DurationalEventTiming {
    pub tick: Option<i32>,
    pub offset: i32,
    pub duration: i32,
}

impl Timing for DurationalEventTiming {
    timing_traits!();

    fn augment_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self {
        if let Some(ref mut t) = self.tick {
            v.augment_target(t);
        }

        v.augment_target(&mut self.duration);
        v.augment_target(&mut self.offset);
        *self
    }

    fn diminish_rhythm<AT: AugDim<i32> + Num + Bounded + Copy>(&mut self, v: AT) -> Self {
        if let Some(ref mut t) = self.tick {
            v.diminish_target(t);
        }

        v.diminish_target(&mut self.duration);
        v.diminish_target(&mut self.offset);
        *self
    }
}

impl DurationalEventTiming {
    pub fn new(duration: i32, tick: Option<i32>, offset: i32) -> Self {
        Self {
            duration,
            tick,
            offset,
        }
    }

    pub fn from_duration(duration: i32) -> Self {
        Self {
            duration,
            tick: None,
            offset: 0,
        }
    }

    pub fn with_duration(&mut self, dur: i32) -> Self {
        self.duration = dur;
        *self
    }

    pub fn mutate_duration(&mut self, f: impl Fn(&mut i32)) -> Self {
        f(&mut self.duration);
        *self
    }

    pub fn end_tick(&self, curr: i32) -> Result<i32> {
        pos_or_err((self.tick.unwrap_or(curr) + self.duration) + self.offset)
    }

    pub fn next_tick(&self, curr: i32) -> Result<i32> {
        Ok(self.tick.unwrap_or(curr) + self.duration)
    }
}

#[macro_export]
/// Macro for generating timing data for a single event.
/// No arguments means no exact tick and no offset.
/// One argument means an optional exact tick.
/// Two arguments mean an optional exact tick and an offset.
macro_rules! timing {
    () => {
        EventTiming::new(None, 0)
    };

    ($tick:expr) => {
        EventTiming::new(Option::from($tick), 0)
    };

    ($tick:expr, $offset:expr) => {
        EventTiming::new(Option::from($tick), i32::from($offset))
    };
}

#[macro_export]
/// Macro for generating timing data for an event that has a duration.
/// One argument means a duration, but no exact tick and no offset.
/// Two argument means a duration and an optional exact tick.
/// Three arguments mean a duration, an optional exact tick and an offset.
macro_rules! duration_with_timing {
    ($dur:expr) => {
        DurationalEventTiming::new($dur, None, 0)
    };

    ($dur:expr, $tick:expr) => {
        DurationalEventTiming::new($dur, Option::from($tick), 0)
    };

    ($dur:expr, $tick:expr, $offset:expr) => {
        DurationalEventTiming::new($dur, Option::from($tick), i32::from($offset))
    };
}

#[cfg(test)]
mod tests {
    use crate::entities::timing::{DurationalEventTiming, EventTiming, Timing};

    #[test]
    fn trait_method_tests_event_timing() {
        let t = timing!();
        assert_eq!(t.tick, None);
        assert_eq!(t.offset, 0);

        let t = timing!(50);
        assert_eq!(t.tick, Some(50));
        assert_eq!(t.offset, 0);

        let t = timing!(None, 50);
        assert_eq!(t.tick, None);
        assert_eq!(t.offset, 50);

        let t = timing!().mutate_exact_tick(|v| *v *= 2);
        assert_eq!(t.tick, None);

        let t = timing!(100).mutate_exact_tick(|v| *v *= 2);
        assert_eq!(t.tick, Some(200));

        let t = timing!(None, -100).mutate_offset(|v| *v = -*v);
        assert_eq!(t.offset, 100);

        let t = timing!(None, 100).with_offset(100);
        assert_eq!(t.start_tick(200).unwrap(), 300);

        let t = timing!(None, -100);
        assert!(t.start_tick(0).is_err());
        assert_eq!(t.start_tick(200).unwrap(), 100);

        let t = timing!(100, 100);
        assert_eq!(t.start_tick(0).unwrap(), 200);
        assert_eq!(t.start_tick(300).unwrap(), 200);
    }

    #[test]
    fn new() {
        assert_eq!(
            EventTiming::new(Some(60), 40),
            EventTiming {
                tick: Some(60),
                offset: 40
            }
        );
        assert_eq!(
            DurationalEventTiming::new(100, Some(60), 40),
            DurationalEventTiming {
                tick: Some(60),
                offset: 40,
                duration: 100
            }
        );
    }

    #[test]
    fn from_duration() {
        assert_eq!(
            DurationalEventTiming::from_duration(100),
            DurationalEventTiming {
                tick: None,
                offset: 0,
                duration: 100,
            },
        );
    }

    #[test]
    fn mutate_duration() {
        let t = duration_with_timing!(150).mutate_duration(|v| *v -= 100);

        assert_eq!(t.duration, 50);
    }

    #[test]
    fn end_tick() {
        let t = duration_with_timing!(50, None, -100);
        assert!(t.end_tick(0).is_err());
        assert_eq!(t.end_tick(200).unwrap(), 150);

        let t = duration_with_timing!(50, 200, -100);
        assert_eq!(t.end_tick(0).unwrap(), 150);
        assert_eq!(t.end_tick(200).unwrap(), 150);

        let t = duration_with_timing!(50, 200, -300);
        assert!(t.end_tick(0).is_err());
        assert!(t.end_tick(2000).is_err());
    }

    #[test]
    fn next_tick() {
        let t = duration_with_timing!(50, None, -100);
        assert_eq!(t.next_tick(0).unwrap(), 50);
        assert_eq!(t.next_tick(200).unwrap(), 250);

        let t = duration_with_timing!(50, 200, -100);
        assert_eq!(t.next_tick(0).unwrap(), 250);
        assert_eq!(t.next_tick(200).unwrap(), 250);

        let t = duration_with_timing!(50, 200, -300);
        assert_eq!(t.next_tick(0).unwrap(), 250);
        assert_eq!(t.next_tick(200).unwrap(), 250);
    }

    #[test]
    fn augment_rhythm() {
        let mut t = timing!(None, -100);
        t.augment_rhythm(2);
        assert_eq!(t, timing!(None, -200));

        let mut t = timing!(200, -100);
        t.augment_rhythm(2);
        assert_eq!(t, timing!(400, -200));

        let mut t = duration_with_timing!(50, None, -100);
        t.augment_rhythm(2);
        assert_eq!(t, duration_with_timing!(100, None, -200));

        let mut t = duration_with_timing!(50, 200, -100);
        t.augment_rhythm(2);
        assert_eq!(t, duration_with_timing!(100, 400, -200));
    }
}
