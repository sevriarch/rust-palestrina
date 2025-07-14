use crate::entities::timing::{EventTiming, Timing};
use anyhow::{anyhow, Result};
use std::{fmt, fmt::Debug};
use thiserror::Error;

fn validate_key_signature(key: &str) -> bool {
    matches!(
        key,
        "Cb" | "Gb"
            | "Db"
            | "Ab"
            | "Eb"
            | "Bb"
            | "F"
            | "C"
            | "G"
            | "D"
            | "A"
            | "E"
            | "B"
            | "F#"
            | "C#"
            | "ab"
            | "eb"
            | "bb"
            | "f"
            | "c"
            | "g"
            | "d"
            | "a"
            | "e"
            | "b"
            | "f#"
            | "c#"
            | "g#"
            | "d#"
            | "a#"
    )
}

fn validate_time_signature(str: &str) -> Result<(u8, u16)> {
    let p: Vec<&str> = str.split("/").collect();

    if p.len() != 2 {
        return Err(anyhow!(MetadataError::InvalidTimeSignature(
            str.to_string()
        )));
    }

    let num = p[0]
        .parse::<u8>()
        .map_err(|_| anyhow!(MetadataError::InvalidTimeSignature(str.to_string())))?;

    let denom = p[1]
        .parse::<u16>()
        .map_err(|_| anyhow!(MetadataError::InvalidTimeSignature(str.to_string())))?;

    if num == 0 || denom & (denom - 1) != 0 {
        Err(anyhow!(MetadataError::InvalidTimeSignature(
            str.to_string()
        )))
    } else {
        Ok((num, denom))
    }
}

fn validate_tempo<T>(t: T) -> Result<f32>
where
    T: TryInto<f32> + Copy + Debug,
{
    t.try_into()
        .map_err(|_| anyhow!(MetadataError::InvalidTempo(format!("{t:?}"))))
        .and_then(|v| {
            if v > 0.0 {
                Ok(v)
            } else {
                Err(anyhow!(MetadataError::InvalidTempo(format!("{t:?}"))))
            }
        })
}

/// A piece of metadata associated with a note or track.
/// Corresponds roughly to a MIDI meta-event or channel event.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataData {
    EndTrack,
    Sustain(bool),
    Tempo(f32),
    KeySignature(String),
    TimeSignature((u8, u16)),
    Instrument(String),
    Text(String),
    Lyric(String),
    Marker(String),
    CuePoint(String),
    Copyright(String),
    TrackName(String),
    Volume(u8),
    Pan(u8),
    Balance(u8),
    PitchBend(f32),
}

impl fmt::Display for MetadataData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match self {
            MetadataData::EndTrack => "EndTrack".to_string(),
            MetadataData::Sustain(v) => format!("Sustain({v})"),
            MetadataData::Tempo(v) => format!("Tempo({v})"),
            MetadataData::KeySignature(v) => format!("KeySignature({v})"),
            MetadataData::TimeSignature(v) => format!("TimeSignature({}/{})", v.0, v.1),
            MetadataData::Instrument(v) => format!("Instrument({v})"),
            MetadataData::Text(v) => format!("Text({v})"),
            MetadataData::Lyric(v) => format!("Lyric({v})"),
            MetadataData::Marker(v) => format!("Marker({v})"),
            MetadataData::CuePoint(v) => format!("CuePoint({v})"),
            MetadataData::Copyright(v) => format!("Copyright({v})"),
            MetadataData::TrackName(v) => format!("TrackName({v})"),
            MetadataData::Volume(v) => format!("Volume({v})"),
            MetadataData::Pan(v) => format!("Pan({v})"),
            MetadataData::Balance(v) => format!("Balance({v})"),
            MetadataData::PitchBend(v) => format!("PitchBend({v})"),
        };

        write!(f, "MetadataData({msg})")
    }
}

impl MetadataData {
    pub fn self_validate(&self) -> Result<()> {
        match self {
            MetadataData::KeySignature(key) => {
                if validate_key_signature(key) {
                    Ok(())
                } else {
                    Err(anyhow!(MetadataError::InvalidKeySignature(key.clone())))
                }
            }
            MetadataData::TimeSignature((n, d)) => {
                if *n < 1 || *d & (*d - 1) != 0 {
                    Err(anyhow!(MetadataError::InvalidTimeSignature(format!(
                        "{n}/{d}"
                    ))))
                } else {
                    Ok(())
                }
            }
            MetadataData::Tempo(t) => {
                if *t > 0.0 {
                    Ok(())
                } else {
                    Err(anyhow!(MetadataError::InvalidTempo(t.to_string())))
                }
            }
            MetadataData::PitchBend(t) => {
                let v = (*t * 4096.0).round() as i16;
                if (-8192..8192).contains(&v) {
                    Ok(())
                } else {
                    Err(anyhow!(MetadataError::InvalidPitchBend(*t)))
                }
            }
            _ => Ok(()),
        }
    }
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
    #[error("Invalid time signature {0}")]
    InvalidTimeSignature(String),
    #[error("Invalid key signature {0}")]
    InvalidKeySignature(String),
    #[error("Invalid tempo {0}")]
    InvalidTempo(String),
    #[error("Invalid pitch bend {0}; absolute value must be less than 2")]
    InvalidPitchBend(f32),
    #[error("Value {0} could not be coerced into a 7-bit integer")]
    Invalid7BitInt(i16),
}

fn metadata_type_exists(event: &str) -> bool {
    matches!(
        event,
        "sustain"
            | "tempo"
            | "key-signature"
            | "time-signature"
            | "instrument"
            | "text"
            | "lyric"
            | "marker"
            | "cue-point"
            | "copyright"
            | "track-name"
            | "volume"
            | "pan"
            | "balance"
            | "pitch-bend"
    )
}

fn get_meta_event_string_data(event: &str, value: String) -> Result<MetadataData> {
    match event {
        "key-signature" => {
            let k = MetadataData::KeySignature(value);

            k.self_validate().map(|_| k)
        }
        "time-signature" => Ok(MetadataData::TimeSignature(validate_time_signature(
            &value,
        )?)),
        "instrument" => Ok(MetadataData::Instrument(value)),
        "text" => Ok(MetadataData::Text(value)),
        "lyric" => Ok(MetadataData::Lyric(value)),
        "marker" => Ok(MetadataData::Marker(value)),
        "cue-point" => Ok(MetadataData::CuePoint(value)),
        "copyright" => Ok(MetadataData::Copyright(value)),
        "trackname" => Ok(MetadataData::TrackName(value)),
        _ => {
            if metadata_type_exists(event) {
                Err(anyhow!(MetadataError::UnexpectedStringValue(
                    event.to_string(),
                    value
                )))
            } else {
                Err(anyhow!(MetadataError::InvalidMetadataType(
                    event.to_string()
                )))
            }
        }
    }
}

fn try_into_7bit_int(val: &i16) -> Result<u8> {
    if (0..=127).contains(val) {
        Ok(*val as u8)
    } else {
        Err(anyhow!(MetadataError::Invalid7BitInt(*val)))
    }
}

fn get_meta_event_int_data(event: &str, value: i16) -> Result<MetadataData> {
    match event {
        "volume" => Ok(MetadataData::Volume(try_into_7bit_int(&value)?)),
        "pan" => Ok(MetadataData::Pan(try_into_7bit_int(&value)?)),
        "balance" => Ok(MetadataData::Balance(try_into_7bit_int(&value)?)),
        "tempo" => Ok(MetadataData::Tempo(validate_tempo(value)?)),
        _ => {
            if metadata_type_exists(event) {
                Err(anyhow!(MetadataError::UnexpectedIntValue(
                    event.to_string(),
                    value
                )))
            } else {
                Err(anyhow!(MetadataError::InvalidMetadataType(
                    event.to_string()
                )))
            }
        }
    }
}

fn get_meta_event_bool_data(event: &str, value: bool) -> Result<MetadataData> {
    match event {
        "sustain" => Ok(MetadataData::Sustain(value)),
        _ => {
            if metadata_type_exists(event) {
                Err(anyhow!(MetadataError::UnexpectedBooleanValue(
                    event.to_string(),
                    value
                )))
            } else {
                Err(anyhow!(MetadataError::InvalidMetadataType(
                    event.to_string()
                )))
            }
        }
    }
}

fn get_meta_event_float_data(event: &str, value: f32) -> Result<MetadataData> {
    match event {
        "tempo" => Ok(MetadataData::Tempo(validate_tempo(value)?)),
        "pitch-bend" => {
            let k = MetadataData::PitchBend(value);
            k.self_validate()?;
            Ok(k)
        }
        _ => {
            if metadata_type_exists(event) {
                Err(anyhow!(MetadataError::UnexpectedFloatValue(
                    event.to_string(),
                    value
                )))
            } else {
                Err(anyhow!(MetadataError::InvalidMetadataType(
                    event.to_string()
                )))
            }
        }
    }
}

/// A chunk of metadata, plus timing data (exact MIDI tick and offset).
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub data: MetadataData,
    pub timing: EventTiming,
}

#[macro_export]
/// Macro for creating a vector of pieces of timed metadata.
///
/// Syntax: metadata_vec![piece1, piece2, piece3...]
///
/// Each piece may be supplied as an untimed piece of metadata (in which case
/// an exact tick of None and an offset of 0 will be applied), or a tuple with
/// three members where the first is an untimed piece of metadata, the second
/// is an exact tick or None, and the third is the offset.
/// ## Example
/// ```
/// use palestrina::entities::timing::EventTiming;
/// use palestrina::metadata::{MetadataData,Metadata,MetadataList};
/// use palestrina::metadata_vec;
///
/// let metadata = MetadataList::new(metadata_vec![
///     // This piece of metadata has an exact tick of None and an offset of 0
///     MetadataData::Text("this text".to_string()),
///     // This piece of metadata has an exact tick of None and an offset of 50
///     (MetadataData::Tempo(160.0), None, 50),
///     // This piece of metadata has an exact tick of 50 and an offset of 100
///     (MetadataData::EndTrack, 50, 100)
/// ]);
/// ```
macro_rules! metadata_vec {
    () => ( Vec::new() );

    (($data:expr, $tick:expr, $offset:expr)) => (
        vec![Metadata {
            data: $data,
            timing: EventTiming{ tick: Option::<i32>::from($tick), offset: $offset }
        }]
    );

    ($data:expr) => (
        vec![Metadata {
            data: $data,
            timing: EventTiming::default()
        }]
    );

    (($data:expr, $tick:expr, $offset:expr), $($tail:tt)*) => (
        {
            let mut ret = vec![Metadata {
                data: $data,
                timing: EventTiming{ tick: Option::<i32>::from($tick), offset: $offset }
            }];

            ret.append(&mut metadata_vec!($($tail)*));
            ret
        }
    );

    ($data:expr, $($tail:tt)*) => (
        {
            let mut ret = vec![Metadata {
                data: $data,
                timing: EventTiming::default()
            }];

            ret.append(&mut metadata_vec!($($tail)*));
            ret
        }
    );
}

#[macro_export]
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
                tick: Option::<i32>::from($tick),
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
        Ok(Metadata {
            data: get_meta_event_string_data(event, value.to_string())?,
            timing: EventTiming::default(),
        })
    }
}

impl TryFrom<(&str, i16)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, i16)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_int_data(event, value)?,
            timing: EventTiming::default(),
        })
    }
}

impl TryFrom<(&str, bool)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, bool)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_bool_data(event, value)?,
            timing: EventTiming::default(),
        })
    }
}

impl TryFrom<(&str, f32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value): (&str, f32)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_float_data(event, value)?,
            timing: EventTiming::default(),
        })
    }
}

impl TryFrom<(&str, &str, Option<i32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, &str, Option<i32>, i32)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_string_data(event, value.to_string())?,
            timing: EventTiming::new(tick, offset),
        })
    }
}

impl TryFrom<(&str, i16, Option<i32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, i16, Option<i32>, i32)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_int_data(event, value)?,
            timing: EventTiming::new(tick, offset),
        })
    }
}

impl TryFrom<(&str, bool, Option<i32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, bool, Option<i32>, i32)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_bool_data(event, value)?,
            timing: EventTiming::new(tick, offset),
        })
    }
}

impl TryFrom<(&str, f32, Option<i32>, i32)> for Metadata {
    type Error = anyhow::Error;

    fn try_from((event, value, tick, offset): (&str, f32, Option<i32>, i32)) -> Result<Self> {
        Ok(Metadata {
            data: get_meta_event_float_data(event, value)?,
            timing: EventTiming::new(tick, offset),
        })
    }
}

impl Metadata {
    /// Create a piece of timed metadata.
    pub fn new(data: MetadataData, tick: Option<i32>, offset: i32) -> Self {
        Self {
            data,
            timing: EventTiming { tick, offset },
        }
    }

    /// Set the exact tick on this piece of timed metadata.
    pub fn with_exact_tick(&mut self, tick: i32) -> &mut Self {
        self.timing.with_exact_tick(tick);
        self
    }

    /// Mutate the exact tick on this piece of timed metadata by passing
    /// a reference to the current exact tick to the supplied closure.
    pub fn mutate_exact_tick(&mut self, f: impl Fn(&mut i32)) -> &mut Self {
        self.timing.mutate_exact_tick(&f);
        self
    }

    /// Set the tick offset on this piece of timed metadata.
    pub fn with_offset(&mut self, offset: i32) -> &mut Self {
        self.timing.with_offset(offset);
        self
    }

    /// Mutate the tick offset on this piece of timed metadata by passing
    /// a reference to the current tick offset to the supplied closure.
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

/// A list of zero or more chunks of timed metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataList {
    pub contents: Vec<Metadata>,
}

impl MetadataList {
    /// Create a new list from a Vec of timed metadata.
    pub fn new(contents: Vec<Metadata>) -> Self {
        MetadataList { contents }
    }

    /// Add a new piece of timed metadata to the list.
    pub fn append(&mut self, md: Metadata) -> &Self {
        self.contents.push(md);
        self
    }

    /// Calculate the latest MIDI tick at which any event associated with
    /// this metadata will appear. The argument is the current tick associated
    /// with this metadata (for metadata associated with a Sequence, this will
    /// be zero, for metadata associated with the start of a note, this will be
    /// the start tick of the note).
    pub fn last_tick(&self, curr: i32) -> Result<i32> {
        if self.contents.is_empty() {
            return Ok(curr);
        }

        let mut max = curr;
        for m in self.contents.iter() {
            let c = m.timing.start_tick(curr)?;

            if c > max {
                max = c;
            }
        }

        Ok(max)
    }

    /// Mutate all exact ticks within this list of metadata by passing each of
    /// them to the supplied closure.
    pub fn mutate_exact_tick(&mut self, f: impl Fn(&mut i32)) -> &Self {
        for m in self.contents.iter_mut() {
            m.timing.mutate_exact_tick(&f);
        }

        self
    }

    /// Mutate all tick offsets within this list of metadata by passing each of
    /// them to the supplied closure.
    pub fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> &Self {
        for m in self.contents.iter_mut() {
            m.timing.mutate_offset(&f);
        }

        self
    }
}

/// Create a new timed metadata object and add it onto the end of a list.
pub trait PushMetadata<T> {
    /// Add a metadata object with default timing (no exact tick, zero offset).
    fn with_metadata(self, tuple: (&str, T)) -> Result<Self>
    where
        Self: Sized;

    /// Add a metadata object with the supplied timing (exact tick, offset).
    fn with_timed_metadata(self, tuple: (&str, T, Option<i32>, i32)) -> Result<Self>
    where
        Self: Sized;
}

macro_rules! push_impl {
    ($($type:ty)*) => ($(
impl PushMetadata<$type> for MetadataList {
    fn with_metadata(mut self, tuple: (&str, $type)) -> Result<Self>
    where
    Self: Sized,
    {
        self.contents.push(Metadata::try_from(tuple)?);
        Ok(self)
    }

    fn with_timed_metadata(mut self, tuple: (&str, $type, Option<i32>, i32)) -> Result<Self>
    where
    Self: Sized,
    {
        self.contents.push(Metadata::try_from(tuple)?);
        Ok(self)
    }
}
    )*)
}

push_impl! { &str i16 f32 bool }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_macro() {
        assert_eq!(
            metadata!(MetadataData::Text("this text".to_string())),
            Metadata {
                data: MetadataData::Text("this text".to_string()),
                timing: EventTiming::default(),
            }
        );

        assert_eq!(
            metadata!(MetadataData::Text("this text".to_string()), None, 100),
            Metadata {
                data: MetadataData::Text("this text".to_string()),
                timing: EventTiming {
                    tick: None,
                    offset: 100
                }
            }
        );

        assert_eq!(
            metadata!(MetadataData::Text("this text".to_string()), 50, 100),
            Metadata {
                data: MetadataData::Text("this text".to_string()),
                timing: EventTiming {
                    tick: Some(50),
                    offset: 100
                }
            }
        );

        assert_eq!(
            metadata_vec![
                MetadataData::Text("this text".to_string()),
                (MetadataData::Tempo(160.0), None, 50),
                (MetadataData::EndTrack, Some(50), 100)
            ],
            vec![
                Metadata {
                    data: MetadataData::Text("this text".to_string()),
                    timing: EventTiming::default(),
                },
                Metadata {
                    data: MetadataData::Tempo(160.0),
                    timing: EventTiming {
                        tick: None,
                        offset: 50
                    },
                },
                Metadata {
                    data: MetadataData::EndTrack,
                    timing: EventTiming {
                        tick: Some(50),
                        offset: 100
                    },
                },
            ]
        );

        assert_eq!(
            metadata_vec![
                (MetadataData::Tempo(160.0), None, 50),
                (MetadataData::EndTrack, Some(50), 100),
                MetadataData::Text("this text".to_string())
            ],
            vec![
                Metadata {
                    data: MetadataData::Tempo(160.0),
                    timing: EventTiming {
                        tick: None,
                        offset: 50
                    },
                },
                Metadata {
                    data: MetadataData::EndTrack,
                    timing: EventTiming {
                        tick: Some(50),
                        offset: 100
                    },
                },
                Metadata {
                    data: MetadataData::Text("this text".to_string()),
                    timing: EventTiming::default(),
                },
            ]
        );
    }

    #[test]
    fn try_from() {
        assert!(Metadata::try_from(("doesn't exist", "really")).is_err());
        assert!(Metadata::try_from(("doesn't exist", "really", Some(50), 100)).is_err());

        assert!(Metadata::try_from(("key-signature", "Vb")).is_err());
        assert_eq!(
            Metadata::try_from(("key-signature", "Eb")).unwrap(),
            metadata!(MetadataData::KeySignature("Eb".to_string()), None, 0)
        );

        assert!(Metadata::try_from(("time-signature", "6:4")).is_err());
        assert!(Metadata::try_from(("time-signature", "-1/4")).is_err());
        assert!(Metadata::try_from(("time-signature", "4/6")).is_err());
        assert_eq!(
            Metadata::try_from(("time-signature", "3/8")).unwrap(),
            metadata!(MetadataData::TimeSignature((3, 8)), None, 0)
        );

        assert!(Metadata::try_from(("volume", -1)).is_err());
        assert!(Metadata::try_from(("volume", 128)).is_err());
        assert_eq!(
            Metadata::try_from(("volume", 0)).unwrap(),
            metadata!(MetadataData::Volume(0), None, 0)
        );
        assert_eq!(
            Metadata::try_from(("volume", 127)).unwrap(),
            metadata!(MetadataData::Volume(127), None, 0)
        );

        assert!(Metadata::try_from(("pan", -1)).is_err());
        assert!(Metadata::try_from(("pan", 128)).is_err());
        assert_eq!(
            Metadata::try_from(("pan", 0)).unwrap(),
            metadata!(MetadataData::Pan(0), None, 0)
        );
        assert_eq!(
            Metadata::try_from(("pan", 127)).unwrap(),
            metadata!(MetadataData::Pan(127), None, 0)
        );

        assert!(Metadata::try_from(("balance", -1)).is_err());
        assert!(Metadata::try_from(("balance", 128)).is_err());
        assert_eq!(
            Metadata::try_from(("balance", 0)).unwrap(),
            metadata!(MetadataData::Balance(0), None, 0)
        );
        assert_eq!(
            Metadata::try_from(("balance", 127)).unwrap(),
            metadata!(MetadataData::Balance(127), None, 0)
        );

        assert!(Metadata::try_from(("tempo", 0)).is_err());
        assert_eq!(
            Metadata::try_from(("tempo", 132)).unwrap(),
            metadata!(MetadataData::Tempo(132.0), None, 0)
        );
        assert_eq!(
            Metadata::try_from(("tempo", 132.0, Some(50), 100)).unwrap(),
            metadata!(MetadataData::Tempo(132.0), Some(50), 100)
        );

        assert_eq!(
            Metadata::try_from(("text", "this text")).unwrap(),
            metadata!(MetadataData::Text("this text".to_string()), None, 0)
        );

        assert_eq!(
            Metadata::try_from(("text", "this text", Some(50), 100)).unwrap(),
            metadata!(MetadataData::Text("this text".to_string()), Some(50), 100)
        );

        assert!(Metadata::try_from(("pitch-bend", 2.0)).is_err());
        assert_eq!(
            Metadata::try_from(("pitch-bend", 1.999)).unwrap(),
            metadata!(MetadataData::PitchBend(1.999), None, 0)
        );

        assert!(Metadata::try_from(("pitch-bend", -2.00013)).is_err());
        assert_eq!(
            Metadata::try_from(("pitch-bend", -2.0, Some(50), 100)).unwrap(),
            metadata!(MetadataData::PitchBend(-2.0), Some(50), 100)
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

    #[test]
    fn with_metadata() {
        assert!(MetadataList::default()
            .with_metadata(("foo", "bar"))
            .is_err());

        assert_eq!(
            MetadataList::default()
                .with_metadata(("text", "test text"))
                .unwrap(),
            MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming::default()
                }]
            }
        );
    }

    #[test]
    fn with_timed_metadata() {
        assert!(MetadataList::default()
            .with_timed_metadata(("foo", "bar", Some(50), 50))
            .is_err());

        assert_eq!(
            MetadataList::default()
                .with_timed_metadata(("text", "test text", Some(50), 50))
                .unwrap(),
            MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(50),
                        offset: 50
                    }
                }]
            }
        );
    }

    #[test]
    fn last_tick() {
        assert_eq!(MetadataList::default().last_tick(0).unwrap(), 0);
        assert_eq!(MetadataList::default().last_tick(128).unwrap(), 128);

        assert_eq!(
            MetadataList {
                contents: vec![
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: None,
                            offset: 50
                        },
                    },
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: Some(100),
                            offset: 25
                        },
                    },
                ],
            }
            .last_tick(0)
            .unwrap(),
            125
        );

        assert_eq!(
            MetadataList {
                contents: vec![
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: None,
                            offset: 50
                        },
                    },
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: Some(100),
                            offset: 50
                        },
                    },
                ],
            }
            .last_tick(200)
            .unwrap(),
            250
        );
    }

    #[test]
    fn list_mutate_exact_tick() {
        assert_eq!(
            MetadataList::default()
                .with_timed_metadata(("text", "test text", None, 50))
                .unwrap()
                .mutate_exact_tick(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: None,
                        offset: 50
                    }
                }]
            }
        );

        assert_eq!(
            MetadataList::default()
                .with_timed_metadata(("text", "test text", Some(20), 50))
                .unwrap()
                .mutate_exact_tick(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(40),
                        offset: 50
                    }
                }]
            }
        );
    }

    #[test]
    fn list_mutate_offset() {
        assert_eq!(
            MetadataList::default()
                .with_timed_metadata(("text", "test text", Some(20), 50))
                .unwrap()
                .mutate_offset(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(20),
                        offset: 100
                    }
                }]
            }
        );
    }
}
