use crate::entities::timing::{EventTiming, Timing};
use anyhow::{anyhow, Result};
use std::fmt::Debug;
use thiserror::Error;

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

#[derive(Clone, Error, Debug)]
pub enum MetadataError {
    #[error("Invalid metadata type found: {0}")]
    InvalidMetadataType(String),
    #[error("Found unexpected string value {1} for metadata type {0}")]
    UnexpectedStringValue(String, String),
    #[error("Found unexpected int value {1} for metadata type {0}")]
    UnexpectedIntValue(String, i16),
    #[error("Found unexpected floating point value {1} for metadata type {0}")]
    UnexpectedFloatValue(String, f32),
    #[error("Found unexpected boolean value {1} for metadata type {0}")]
    UnexpectedBooleanValue(String, bool),
}

fn get_meta_event_string_data(event: &str, value: String) -> Result<MetadataData> {
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
        _ => Err(anyhow!(MetadataError::UnexpectedStringValue(
            event.to_string(),
            value
        ))),
    }
}

fn get_meta_event_int_data(event: &str, value: i16) -> Result<MetadataData> {
    match event {
        "volume" => Ok(MetadataData::Volume(value)),
        "pan" => Ok(MetadataData::Pan(value)),
        "balance" => Ok(MetadataData::Balance(value)),
        "pitch-bend" => Ok(MetadataData::PitchBend(value)),
        "tempo" => Ok(MetadataData::Tempo(value as f32)),
        _ => Err(anyhow!(MetadataError::UnexpectedIntValue(
            event.to_string(),
            value
        ))),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub data: MetadataData,
    pub timing: EventTiming,
}

macro_rules! metadata {
    ($data:expr) => {
        Metadata {
            data: $data,
            timing: EventTiming::default(),
        }
    };

    ($data:expr, $tick:expr, $offset:expr) => {
        Metadata {
            data: $data,
            timing: EventTiming {
                tick: $tick,
                offset: $offset,
            },
        }
    };
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
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, &str)) -> Result<Self> {
        let data = get_meta_event_string_data(event, value.to_string())?;

        Ok(metadata!(data))
    }
}

impl TryFrom<(&str, i16)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, i16)) -> Result<Self> {
        let data = get_meta_event_int_data(event, value)?;

        Ok(metadata!(data))
    }
}

impl TryFrom<(&str, bool)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, bool)) -> Result<Self> {
        if event == "sustain" {
            Ok(metadata!(MetadataData::Sustain(value)))
        } else {
            Err(anyhow!(MetadataError::UnexpectedBooleanValue(
                event.to_string(),
                value
            )))
        }
    }
}

impl TryFrom<(&str, f32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, f32)) -> Result<Self> {
        if event == "tempo" {
            Ok(metadata!(MetadataData::Tempo(value)))
        } else {
            Err(anyhow!(MetadataError::UnexpectedFloatValue(
                event.to_string(),
                value,
            )))
        }
    }
}

impl TryFrom<(&str, &str, Option<u32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, &str, Option<u32>, i32)) -> Result<Self> {
        let data = get_meta_event_string_data(event, value.to_string())?;

        Ok(metadata!(data, tick, offset))
    }
}

impl TryFrom<(&str, i16, Option<u32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, i16, Option<u32>, i32)) -> Result<Self> {
        let data = get_meta_event_int_data(event, value)?;

        Ok(metadata!(data, tick, offset))
    }
}

impl TryFrom<(&str, bool, Option<u32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, bool, Option<u32>, i32)) -> Result<Self> {
        if event == "sustain" {
            Ok(metadata!(MetadataData::Sustain(value), tick, offset))
        } else {
            Err(anyhow!(MetadataError::UnexpectedBooleanValue(
                event.to_string(),
                value
            )))
        }
    }
}

impl TryFrom<(&str, f32, Option<u32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, f32, Option<u32>, i32)) -> Result<Self> {
        if event == "tempo" {
            Ok(metadata!(MetadataData::Tempo(value), tick, offset))
        } else {
            Err(anyhow!(MetadataError::UnexpectedFloatValue(
                event.to_string(),
                value,
            )))
        }
    }
}

impl Metadata {
    pub fn new(data: MetadataData, tick: Option<u32>, offset: i32) -> Self {
        Self {
            data,
            timing: EventTiming { tick, offset },
        }
    }

    pub fn with_exact_tick(&mut self, tick: u32) -> &mut Self {
        self.timing.with_exact_tick(tick);
        self
    }

    pub fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> &mut Self {
        self.timing.mutate_exact_tick(&f);
        self
    }

    pub fn with_offset(&mut self, offset: i32) -> &mut Self {
        self.timing.with_offset(offset);
        self
    }

    pub fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> &mut Self {
        self.timing.mutate_offset(&f);
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
    fn try_from() {
        assert!(Metadata::try_from(("doesn't exist", "really")).is_err());
        assert!(Metadata::try_from(("doesn't exist", "really", Some(50), 100)).is_err());

        assert_eq!(
            Metadata::try_from(("text", "this text")).unwrap(),
            metadata!(MetadataData::Text("this text".to_string()), None, 0)
        );

        assert_eq!(
            Metadata::try_from(("text", "this text", Some(50), 100)).unwrap(),
            metadata!(MetadataData::Text("this text".to_string()), Some(50), 100)
        );

        assert_eq!(
            Metadata::try_from(("pitch-bend", 4096)).unwrap(),
            metadata!(MetadataData::PitchBend(4096), None, 0)
        );

        assert_eq!(
            Metadata::try_from(("pitch-bend", 4096, Some(50), 100)).unwrap(),
            metadata!(MetadataData::PitchBend(4096), Some(50), 100)
        );

        assert_eq!(
            Metadata::try_from(("tempo", 132)).unwrap(),
            metadata!(MetadataData::Tempo(132.0), None, 0)
        );

        assert_eq!(
            Metadata::try_from(("tempo", 132.0, Some(50), 100)).unwrap(),
            metadata!(MetadataData::Tempo(132.0), Some(50), 100)
        );

        assert_eq!(
            Metadata::try_from(("sustain", true)).unwrap(),
            metadata!(MetadataData::Sustain(true), None, 0)
        );

        assert_eq!(
            Metadata::try_from(("sustain", true, Some(50), 100)).unwrap(),
            metadata!(MetadataData::Sustain(true), Some(50), 100)
        )
    }

    #[test]
    fn with_exact_tick() {
        assert_eq!(
            Metadata::default().with_exact_tick(500),
            &metadata!(MetadataData::EndTrack, Some(500), 0)
        );
    }

    #[test]
    fn mutate_exact_tick() {
        assert_eq!(
            Metadata::default().mutate_exact_tick(|v| *v += 50),
            &metadata!(MetadataData::EndTrack, None, 0)
        );

        assert_eq!(
            Metadata::default()
                .with_exact_tick(50)
                .mutate_exact_tick(|v| *v += 50),
            &metadata!(MetadataData::EndTrack, Some(100), 0)
        );
    }

    #[test]
    fn with_offset() {
        assert_eq!(
            Metadata::default().with_offset(50),
            &metadata!(MetadataData::EndTrack, None, 50)
        );
    }

    #[test]
    fn mutate_offset() {
        assert_eq!(
            Metadata::default()
                .with_offset(50)
                .mutate_offset(|v| *v += 50),
            &metadata!(MetadataData::EndTrack, None, 100)
        );
    }
}
