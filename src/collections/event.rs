use crate::collections::traits::Collection;
use crate::default_methods;
use std::fmt::Debug;

pub trait Timing {
    fn with_exact_tick(self, tick: i32) -> Self;
    fn with_offset(self, offset: i32) -> Self;
    fn calculate_tick(&self, curr: i32) -> i32;
}

#[derive(Copy, Clone, Debug)]
pub struct EventTiming {
    tick: Option<i32>,
    offset: i32,
}

impl EventTiming {
    fn with_exact_tick(mut self, tick: i32) -> Self {
        self.tick = Some(tick);
        self
    }

    fn with_offset(mut self, offset: i32) -> Self {
        self.offset = offset;
        self
    }

    fn calculate_tick(&self, curr: i32) -> i32 {
        self.offset
            + if let Some(exact) = self.tick {
                exact
            } else {
                curr
            }
    }
}

#[derive(Copy, Clone, Debug)]
enum MetaEventValue<'a> {
    StringValue(&'a str),
    NumValue(i32),
}

#[derive(Copy, Clone, Debug)]
pub struct MetaEvent<'a> {
    kind: &'a str,
    value: MetaEventValue<'a>,
    timing: EventTiming,
}

impl<'a> MetaEvent<'a> {
    fn new(kind: &'a str, value: MetaEventValue<'a>, tick: Option<i32>, offset: i32) -> Self {
        Self {
            kind,
            value,
            timing: EventTiming { tick, offset },
        }
    }
}

struct EventList<'a> {
    contents: Vec<MetaEvent<'a>>,
}

impl<'a> Collection<MetaEvent<'a>> for EventList<'a> {
    default_methods!(MetaEvent<'a>);
}

impl<'a> EventList<'a> {
    pub fn augment_rhythm(mut self, by: i32) -> Result<Self, String> {
        for e in self.contents.iter_mut() {
            e.timing.offset *= by;
        }

        Ok(self)
    }

    pub fn diminish_rhythm(mut self, by: i32) -> Result<Self, String> {
        if by == 0 {
            return Err("cannot diminish by 0".to_string());
        }

        for e in self.contents.iter_mut() {
            e.timing.offset /= by;
        }

        Ok(self)
    }
}
