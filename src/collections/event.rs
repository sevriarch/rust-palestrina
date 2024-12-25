use crate::collections::traits::Collection;
use crate::default_methods;
use crate::entities::timing::{EventTiming, Timing};
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub enum MetaEventData {
    EndTrack,
    Sustain(bool),
    Tempo(f32),
    KeySignature(String),
    TimeSignature(String),
    Instrument(String),
    Text(String),
    Lyric(String),
    Marker(String),
    CuePoint(String),
    Copyright(String),
    TrackName(String),
    Volume(i16),
    Pan(i16),
    Balance(i16),
    PitchBend(i16),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetaEvent {
    data: MetaEventData,
    timing: EventTiming,
}

impl MetaEvent {
    pub fn new(data: MetaEventData, tick: Option<u32>, offset: i32) -> Self {
        Self {
            data,
            timing: EventTiming { tick, offset },
        }
    }

    pub fn with_exact_tick(mut self, tick: u32) -> Self {
        self.timing = self.timing.with_exact_tick(tick);
        self
    }

    pub fn with_offset(mut self, offset: i32) -> Self {
        self.timing = self.timing.with_offset(offset);
        self
    }
}

impl Default for MetaEvent {
    fn default() -> MetaEvent {
        Self {
            data: MetaEventData::EndTrack,
            timing: EventTiming::default(),
        }
    }
}

#[derive(Debug)]
pub enum MetaEventError {
    InvalidEventType,
    InvalidValueType,
}

fn get_meta_event_string_data(event: &str, value: String) -> Result<MetaEventData, MetaEventError> {
    match event {
        "key-signature" => Ok(MetaEventData::KeySignature(value)),
        "time-signature" => Ok(MetaEventData::TimeSignature(value)),
        "instrument" => Ok(MetaEventData::Instrument(value)),
        "text" => Ok(MetaEventData::Text(value)),
        "lyric" => Ok(MetaEventData::Lyric(value)),
        "marker" => Ok(MetaEventData::Marker(value)),
        "cue-point" => Ok(MetaEventData::CuePoint(value)),
        "copyright" => Ok(MetaEventData::Copyright(value)),
        "trackname" => Ok(MetaEventData::TrackName(value)),
        _ => Err(MetaEventError::InvalidValueType),
    }
}

fn get_meta_event_int_data(event: &str, value: i16) -> Result<MetaEventData, MetaEventError> {
    match event {
        "volume" => Ok(MetaEventData::Volume(value)),
        "pan" => Ok(MetaEventData::Pan(value)),
        "balance" => Ok(MetaEventData::Balance(value)),
        "pitch-bend" => Ok(MetaEventData::PitchBend(value)),
        "tempo" => Ok(MetaEventData::Tempo(value as f32)),
        _ => Err(MetaEventError::InvalidValueType),
    }
}

impl From<MetaEventData> for MetaEvent {
    fn from(data: MetaEventData) -> Self {
        MetaEvent {
            data,
            timing: EventTiming::default(),
        }
    }
}

impl TryFrom<(&str, &str)> for MetaEvent {
    type Error = MetaEventError;

    fn try_from((event, value): (&str, &str)) -> Result<Self, Self::Error> {
        let data = get_meta_event_string_data(event, value.to_string())?;

        Ok(MetaEvent::from(data))
    }
}

impl TryFrom<(&str, i16)> for MetaEvent {
    type Error = MetaEventError;

    fn try_from((event, value): (&str, i16)) -> Result<Self, Self::Error> {
        let data = get_meta_event_int_data(event, value)?;

        Ok(MetaEvent::from(data))
    }
}

impl TryFrom<(&str, bool)> for MetaEvent {
    type Error = MetaEventError;

    fn try_from((event, value): (&str, bool)) -> Result<Self, Self::Error> {
        if event == "sustain" {
            Ok(MetaEvent::from(MetaEventData::Sustain(value)))
        } else {
            Err(MetaEventError::InvalidValueType)
        }
    }
}

impl TryFrom<(&str, f32)> for MetaEvent {
    type Error = MetaEventError;

    fn try_from((event, value): (&str, f32)) -> Result<Self, Self::Error> {
        if event == "tempo" {
            Ok(MetaEvent::from(MetaEventData::Tempo(value)))
        } else {
            Err(MetaEventError::InvalidValueType)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventList {
    contents: Vec<MetaEvent>,
}

impl Collection<MetaEvent> for EventList {
    default_methods!(MetaEvent);
}

impl EventList {
    pub fn new(contents: Vec<MetaEvent>) -> Self {
        EventList { contents }
    }

    pub fn augment_rhythm(self, by: i32) -> Result<Self, String> {
        Ok(self.mutate_each(|m| m.timing.offset *= by))
    }

    pub fn diminish_rhythm(self, by: i32) -> Result<Self, String> {
        if by == 0 {
            return Err("cannot diminish by 0".to_string());
        }

        Ok(self.mutate_each(|m| m.timing.offset /= by))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_exact_tick() {
        assert_eq!(
            MetaEvent::default().with_exact_tick(500),
            MetaEvent {
                data: MetaEventData::EndTrack,
                timing: EventTiming {
                    tick: Some(500),
                    offset: 0
                }
            }
        );
    }

    #[test]
    fn with_offset() {
        assert_eq!(
            MetaEvent::default().with_offset(50),
            MetaEvent {
                data: MetaEventData::EndTrack,
                timing: EventTiming {
                    tick: None,
                    offset: 50
                }
            }
        );
    }

    #[test]
    fn try_from() {
        assert!(MetaEvent::try_from(("doesn't exist", "really")).is_err());

        assert_eq!(
            MetaEvent::try_from(("text", "this text")).unwrap().data,
            MetaEventData::Text("this text".to_string())
        );

        assert_eq!(
            MetaEvent::try_from(("pitch-bend", 4096)).unwrap().data,
            MetaEventData::PitchBend(4096)
        );

        assert_eq!(
            MetaEvent::try_from(("tempo", 132)).unwrap().data,
            MetaEventData::Tempo(132.0)
        );

        assert_eq!(
            MetaEvent::try_from(("tempo", 132.0)).unwrap().data,
            MetaEventData::Tempo(132.0)
        );

        assert_eq!(
            MetaEvent::try_from(("sustain", true)).unwrap().data,
            MetaEventData::Sustain(true)
        )
    }

    #[test]
    fn augment_rhythm() {
        assert_eq!(
            EventList::new(vec![MetaEvent::try_from(("sustain", true))
                .unwrap()
                .with_offset(100)])
            .augment_rhythm(3)
            .unwrap(),
            EventList::new(vec![MetaEvent::try_from(("sustain", true))
                .unwrap()
                .with_offset(300)])
        );
    }

    #[test]
    fn diminish_rhythm() {
        assert_eq!(
            EventList::new(vec![MetaEvent::try_from(("sustain", true))
                .unwrap()
                .with_offset(300)])
            .diminish_rhythm(3)
            .unwrap(),
            EventList::new(vec![MetaEvent::try_from(("sustain", true))
                .unwrap()
                .with_offset(100)])
        );
    }
}
