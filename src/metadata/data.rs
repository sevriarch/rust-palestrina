use crate::entities::timing::{EventTiming, Timing};
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataData {
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

#[derive(Debug)]
pub enum MetadataError {
    InvalidEventType,
    InvalidValueType,
}

fn get_meta_event_string_data(event: &str, value: String) -> Result<MetadataData, MetadataError> {
    match event {
        "key-signature" => Ok(MetadataData::KeySignature(value)),
        "time-signature" => Ok(MetadataData::TimeSignature(value)),
        "instrument" => Ok(MetadataData::Instrument(value)),
        "text" => Ok(MetadataData::Text(value)),
        "lyric" => Ok(MetadataData::Lyric(value)),
        "marker" => Ok(MetadataData::Marker(value)),
        "cue-point" => Ok(MetadataData::CuePoint(value)),
        "copyright" => Ok(MetadataData::Copyright(value)),
        "trackname" => Ok(MetadataData::TrackName(value)),
        _ => Err(MetadataError::InvalidValueType),
    }
}

fn get_meta_event_int_data(event: &str, value: i16) -> Result<MetadataData, MetadataError> {
    match event {
        "volume" => Ok(MetadataData::Volume(value)),
        "pan" => Ok(MetadataData::Pan(value)),
        "balance" => Ok(MetadataData::Balance(value)),
        "pitch-bend" => Ok(MetadataData::PitchBend(value)),
        "tempo" => Ok(MetadataData::Tempo(value as f32)),
        _ => Err(MetadataError::InvalidValueType),
    }
}

impl From<MetadataData> for Metadata {
    fn from(data: MetadataData) -> Self {
        Metadata {
            data,
            timing: EventTiming::default(),
        }
    }
}

impl TryFrom<(&str, &str)> for Metadata {
    type Error = MetadataError;

    fn try_from((event, value): (&str, &str)) -> Result<Self, Self::Error> {
        let data = get_meta_event_string_data(event, value.to_string())?;

        Ok(Metadata::from(data))
    }
}

impl TryFrom<(&str, i16)> for Metadata {
    type Error = MetadataError;

    fn try_from((event, value): (&str, i16)) -> Result<Self, Self::Error> {
        let data = get_meta_event_int_data(event, value)?;

        Ok(Metadata::from(data))
    }
}

impl TryFrom<(&str, bool)> for Metadata {
    type Error = MetadataError;

    fn try_from((event, value): (&str, bool)) -> Result<Self, Self::Error> {
        if event == "sustain" {
            Ok(Metadata::from(MetadataData::Sustain(value)))
        } else {
            Err(MetadataError::InvalidValueType)
        }
    }
}

impl TryFrom<(&str, f32)> for Metadata {
    type Error = MetadataError;

    fn try_from((event, value): (&str, f32)) -> Result<Self, Self::Error> {
        if event == "tempo" {
            Ok(Metadata::from(MetadataData::Tempo(value)))
        } else {
            Err(MetadataError::InvalidValueType)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub data: MetadataData,
    pub timing: EventTiming,
}

impl Metadata {
    pub fn new(data: MetadataData, tick: Option<u32>, offset: i32) -> Self {
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

    pub fn mutate_offset(mut self, f: fn(&mut i32)) -> Self {
        f(&mut self.timing.offset);
        self
    }
}

impl Default for Metadata {
    fn default() -> Metadata {
        Self {
            data: MetadataData::EndTrack,
            timing: EventTiming::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_exact_tick() {
        assert_eq!(
            Metadata::default().with_exact_tick(500),
            Metadata {
                data: MetadataData::EndTrack,
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
            Metadata::default().with_offset(50),
            Metadata {
                data: MetadataData::EndTrack,
                timing: EventTiming {
                    tick: None,
                    offset: 50
                }
            }
        );
    }

    #[test]
    fn mutate_offset() {
        assert_eq!(
            Metadata::default()
                .with_offset(50)
                .mutate_offset(|v| *v += 50),
            Metadata {
                data: MetadataData::EndTrack,
                timing: EventTiming {
                    tick: None,
                    offset: 100
                }
            }
        );
    }

    #[test]
    fn try_from() {
        assert!(Metadata::try_from(("doesn't exist", "really")).is_err());

        assert_eq!(
            Metadata::try_from(("text", "this text")).unwrap().data,
            MetadataData::Text("this text".to_string())
        );

        assert_eq!(
            Metadata::try_from(("pitch-bend", 4096)).unwrap().data,
            MetadataData::PitchBend(4096)
        );

        assert_eq!(
            Metadata::try_from(("tempo", 132)).unwrap().data,
            MetadataData::Tempo(132.0)
        );

        assert_eq!(
            Metadata::try_from(("tempo", 132.0)).unwrap().data,
            MetadataData::Tempo(132.0)
        );

        assert_eq!(
            Metadata::try_from(("sustain", true)).unwrap().data,
            MetadataData::Sustain(true)
        )
    }

    /*
    #[test]
    fn augment_rhythm() {
        assert_eq!(
            EventList::new(vec![Metadata::try_from(("sustain", true))
                .unwrap()
                .with_offset(100)])
            .augment_rhythm(3)
            .unwrap(),
            EventList::new(vec![Metadata::try_from(("sustain", true))
                .unwrap()
                .with_offset(300)])
        );
    }

    #[test]
    fn diminish_rhythm() {
        assert_eq!(
            EventList::new(vec![Metadata::try_from(("sustain", true))
                .unwrap()
                .with_offset(300)])
            .diminish_rhythm(3)
            .unwrap(),
            EventList::new(vec![Metadata::try_from(("sustain", true))
                .unwrap()
                .with_offset(100)])
        );
    }
    */
}
