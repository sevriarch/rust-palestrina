pub trait Timing {
    fn with_exact_tick(&mut self, tick: u32) -> Self;
    fn with_offset(&mut self, tick: i32) -> Self;
    fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> Self;
    fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> Self;
    fn start_tick(&self, curr: u32) -> Result<u32, String>;
}

fn exact_or_curr(exact: Option<u32>, curr: u32) -> u32 {
    if let Some(tick) = exact {
        tick
    } else {
        curr
    }
}

fn pos_or_err(tick: i32) -> Result<u32, String> {
    if tick < 0 {
        Err(format!("negative tick {:?} is not permitted", tick))
    } else {
        Ok(tick as u32)
    }
}

macro_rules! timing_traits {
    ($type:ty) => {
        impl Timing for $type {
            fn with_exact_tick(&mut self, tick: u32) -> Self {
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

            fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> Self {
                if let Some(tick) = self.tick.as_mut() {
                    f(tick);
                }
                *self
            }

            fn start_tick(&self, curr: u32) -> Result<u32, String> {
                pos_or_err(self.offset + exact_or_curr(self.tick, curr) as i32)
            }
        }
    };
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EventTiming {
    pub tick: Option<u32>,
    pub offset: i32,
}

timing_traits!(EventTiming);

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DurationalEventTiming {
    pub tick: Option<u32>,
    pub offset: i32,
    pub duration: u32,
}

timing_traits!(DurationalEventTiming);

impl DurationalEventTiming {
    pub fn with_duration(&mut self, dur: u32) -> Self {
        self.duration = dur;
        *self
    }

    pub fn mutate_duration(&mut self, f: impl Fn(&mut u32)) -> Self {
        f(&mut self.duration);
        *self
    }

    pub fn end_tick(&self, curr: u32) -> Result<u32, String> {
        pos_or_err((exact_or_curr(self.tick, curr) + self.duration) as i32 + self.offset)
    }

    pub fn next_tick(&self, curr: u32) -> Result<u32, String> {
        Ok(exact_or_curr(self.tick, curr) + self.duration)
    }

    pub fn augment_rhythm(mut self, v: u32) -> Result<Self, String> {
        self.duration *= v;
        self.offset *= v as i32;
        Ok(self)
    }
}

#[cfg(test)]
mod test_event_timing {
    use crate::entities::timing::{DurationalEventTiming, EventTiming, Timing};

    macro_rules! trait_method_tests {
        ($type:ty) => {
            let t = <$type>::default();
            assert_eq!(t.tick, None);
            assert_eq!(t.offset, 0);

            let t = <$type>::default().with_exact_tick(50);
            assert_eq!(t.tick, Some(50));

            let t = <$type>::default().with_offset(50);
            assert_eq!(t.offset, 50);

            let t = <$type>::default().mutate_exact_tick(|v| *v *= 2);
            assert_eq!(t.tick, None);

            let t = <$type>::default()
                .with_exact_tick(100)
                .mutate_exact_tick(|v| *v *= 2);
            assert_eq!(t.tick, Some(200));

            let t = <$type>::default()
                .with_offset(-100)
                .mutate_offset(|v| *v = -*v);
            assert_eq!(t.offset, 100);

            let t = <$type>::default().with_offset(100);
            assert_eq!(t.start_tick(200), Ok(300));

            let t = <$type>::default().with_offset(-100);
            assert!(t.start_tick(0).is_err());
            assert_eq!(t.start_tick(200), Ok(100));

            let t = <$type>::default().with_exact_tick(300).with_offset(-100);
            assert_eq!(t.start_tick(0), Ok(200));
            assert_eq!(t.start_tick(300), Ok(200));
        };
    }

    #[test]
    fn trait_method_tests_event_timing() {
        trait_method_tests!(EventTiming);
    }

    #[test]
    fn trait_method_tests_durational() {
        trait_method_tests!(DurationalEventTiming);
    }

    #[test]
    fn mutate_duration() {
        let t = DurationalEventTiming::default()
            .with_duration(150)
            .mutate_duration(|v| *v -= 100);

        assert_eq!(t.duration, 50);
    }

    #[test]
    fn end_tick() {
        let t = DurationalEventTiming::default()
            .with_offset(-100)
            .with_duration(50);

        assert!(t.end_tick(0).is_err());
        assert_eq!(t.end_tick(200), Ok(150));

        let t = DurationalEventTiming::default()
            .with_exact_tick(200)
            .with_offset(-100)
            .with_duration(50);

        assert_eq!(t.end_tick(0), Ok(150));
        assert_eq!(t.end_tick(200), Ok(150));

        let t = DurationalEventTiming::default()
            .with_exact_tick(200)
            .with_offset(-300)
            .with_duration(50);

        assert!(t.end_tick(0).is_err());
        assert!(t.end_tick(2000).is_err());
    }

    #[test]
    fn next_tick() {
        let t = DurationalEventTiming::default()
            .with_offset(-100)
            .with_duration(50);

        assert_eq!(t.next_tick(0), Ok(50));
        assert_eq!(t.next_tick(200), Ok(250));

        let t = DurationalEventTiming::default()
            .with_exact_tick(200)
            .with_offset(-100)
            .with_duration(50);

        assert_eq!(t.next_tick(0), Ok(250));
        assert_eq!(t.next_tick(200), Ok(250));

        let t = DurationalEventTiming::default()
            .with_exact_tick(200)
            .with_offset(-300)
            .with_duration(50);

        assert_eq!(t.next_tick(0), Ok(250));
        assert_eq!(t.next_tick(200), Ok(250));
    }
}
