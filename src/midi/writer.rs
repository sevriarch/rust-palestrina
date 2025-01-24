use crate::collections::traits::Collection;
use crate::entities::timing::Timing;
use crate::metadata::{Metadata, MetadataData, MetadataError, MetadataList};
use crate::score::Score;
use crate::sequences::melody::{Melody, MelodyMember};
use anyhow::{anyhow, Result};
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use thiserror::Error;

use super::constants::*;

#[derive(Clone, Error, Debug)]
pub enum MidiError {
    #[error("Invalid number of bytes requested: {0}")]
    InvalidRequestedBytes(usize),
    #[error("Number {0} does not fit in requested number of bytes {1}")]
    NumberExceedsRequestedBytes(u32, usize),
    #[error("Number {0} exceeds maximum permitted {1}")]
    NumberExceedsMaximumPermitted(u32, u32),
    #[error("Pitch {0} out of permitted range (0x00-0x7f)")]
    PitchOutOfRange(String),
    #[error("Unsupported metadata {0}")]
    UnsupportedMetadata(MetadataData),
}

pub trait ToMidiBytes {
    fn try_to_midi_bytes(&self) -> Result<Vec<u8>>;
}

pub trait WriteMidiBytes {
    fn try_to_hash(&self) -> Result<u64>;
    fn try_to_write_midi_bytes(&self, file: &str) -> Result<()>;
}

pub trait ToTimedMidiBytes {
    fn try_to_timed_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>)>;
}

pub trait ToVecTimedMidiBytes {
    fn try_to_vec_timed_midi_bytes(&self, curr: u32) -> Result<Vec<(u32, Vec<u8>)>>;
}

fn try_number_to_fixed_bytes(num: u32, size: usize) -> Result<Vec<u8>> {
    if size == 0 || size > 4 {
        return Err(anyhow!(MidiError::InvalidRequestedBytes(size)));
    }

    let mut bytes = Vec::<u8>::with_capacity(size);
    let mut n = num;
    bytes.push((n & 0xff) as u8);

    for _ in 1..size {
        n >>= 8;
        bytes.push((n & 0xff) as u8);
    }

    if n > 0xff {
        return Err(anyhow!(MidiError::NumberExceedsRequestedBytes(n, size)));
    }

    bytes.reverse();
    Ok(bytes)
}

fn try_number_to_variable_bytes(num: u32) -> Result<Vec<u8>> {
    if num > 0xfffffff {
        return Err(anyhow!(MidiError::NumberExceedsMaximumPermitted(
            num, 0xfffffff
        )));
    }

    let mut bytes = Vec::<u8>::with_capacity(4);
    let mut n = num;
    bytes.push((n & 0x7f) as u8);

    while n > 0x7f {
        n >>= 7;
        bytes.push((n & 0x7f | 0x80) as u8);
    }

    bytes.reverse();
    Ok(bytes)
}

fn try_string_to_variable_bytes(str: &str) -> Result<Vec<u8>> {
    let mut chars: Vec<u8> = str
        .chars()
        .flat_map(|c| c.to_string().into_bytes())
        .collect();
    let mut ret = try_number_to_variable_bytes(chars.len() as u32)?;

    ret.append(&mut chars);

    Ok(ret)
}

fn key_signature_to_midi_bytes(key: &str) -> Result<Vec<u8>> {
    let byte1 = match key {
        "C" => Ok(0),
        "a" => Ok(0),
        "G" => Ok(1),
        "e" => Ok(1),
        "D" => Ok(2),
        "b" => Ok(2),
        "A" => Ok(3),
        "f#" => Ok(3),
        "E" => Ok(4),
        "c#" => Ok(4),
        "B" => Ok(5),
        "g#" => Ok(5),
        "F#" => Ok(6),
        "d#" => Ok(6),
        "C#" => Ok(7),
        "a#" => Ok(7),
        "Cb" => Ok(249),
        "ab" => Ok(249),
        "Gb" => Ok(250),
        "eb" => Ok(250),
        "Db" => Ok(251),
        "bb" => Ok(251),
        "Ab" => Ok(252),
        "f" => Ok(252),
        "Eb" => Ok(253),
        "c" => Ok(253),
        "Bb" => Ok(254),
        "g" => Ok(254),
        "F" => Ok(255),
        "d" => Ok(255),
        _ => Err(MetadataError::InvalidKeySignature(key.to_string())),
    }?;

    let mut byte2 = 0x00;
    if let Some(first) = key.chars().next() {
        if first.is_lowercase() {
            byte2 = 0x01;
        }
    }

    Ok(vec![0xff, 0x59, 0x02, byte1, byte2])
}

fn time_signature_to_midi_bytes(v: &(u8, u16)) -> Vec<u8> {
    vec![
        0xff,
        0x58,
        0x04,
        v.0,
        15 - v.1.leading_zeros() as u8,
        0x18,
        0x08,
    ]
}

fn controller_event(v: u8, b: u8) -> Vec<u8> {
    vec![CONTROLLER_BYTE, v, b]
}

fn sustain_to_midi_bytes(v: &bool) -> Vec<u8> {
    controller_event(
        SUSTAIN_CONTROLLER,
        if *v { EVENT_ON_VALUE } else { EVENT_OFF_VALUE },
    )
}

fn pitch_bend_to_midi_bytes(f: &f32) -> Result<Vec<u8>> {
    let v = (4096.0 * *f).round() as i32 - 8192;

    if (-16384..0).contains(&v) {
        Ok(vec![0xe0, (v & 0x7f) as u8, (v >> 7 & 0x7f) as u8])
    } else {
        println!("v is {}, f is {}", v, f);
        Err(anyhow!(MetadataError::InvalidPitchBend(*f)))
    }
}

macro_rules! build_text_event {
    ($type:expr, $txt:expr) => {
        Ok($type
            .iter()
            .chain(try_string_to_variable_bytes($txt)?.iter())
            .cloned()
            .collect())
    };
}

impl ToMidiBytes for MetadataData {
    fn try_to_midi_bytes(&self) -> Result<Vec<u8>> {
        // Ensure that the metadata is actually valid before attempting to convert
        self.self_validate()?;

        match self {
            MetadataData::EndTrack => Ok(END_TRACK_EVENT.to_vec()),
            MetadataData::Tempo(t) => {
                let mut ret = TEMPO_EVENT.to_vec();
                ret.append(&mut try_number_to_fixed_bytes((6e7 / t).round() as u32, 3)?);

                Ok(ret)
            }
            MetadataData::Text(txt) => build_text_event!(TEXT_EVENT, txt),
            MetadataData::Copyright(txt) => build_text_event!(COPYRIGHT_EVENT, txt),
            MetadataData::TrackName(txt) => build_text_event!(TRACK_NAME_EVENT, txt),
            MetadataData::Instrument(txt) => build_text_event!(INSTRUMENT_NAME_EVENT, txt),
            MetadataData::Lyric(txt) => build_text_event!(LYRIC_EVENT, txt),
            MetadataData::Marker(txt) => build_text_event!(MARKER_EVENT, txt),
            MetadataData::CuePoint(txt) => build_text_event!(CUE_POINT_EVENT, txt),
            MetadataData::Sustain(val) => Ok(sustain_to_midi_bytes(val)),
            MetadataData::KeySignature(val) => Ok(key_signature_to_midi_bytes(val)?),
            MetadataData::TimeSignature(val) => Ok(time_signature_to_midi_bytes(val)),
            MetadataData::PitchBend(val) => Ok(pitch_bend_to_midi_bytes(val)?),
            MetadataData::Volume(val) => Ok(controller_event(VOLUME_CONTROLLER, *val)),
            MetadataData::Pan(val) => Ok(controller_event(PAN_CONTROLLER, *val)),
            MetadataData::Balance(val) => Ok(controller_event(BALANCE_CONTROLLER, *val)),
        }
    }
}

impl ToTimedMidiBytes for Metadata {
    fn try_to_timed_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>)> {
        let databytes = self.data.try_to_midi_bytes()?;

        Ok((self.timing.start_tick(curr)?, databytes))
    }
}

impl ToVecTimedMidiBytes for MetadataList {
    fn try_to_vec_timed_midi_bytes(&self, curr: u32) -> Result<Vec<(u32, Vec<u8>)>> {
        self.contents
            .iter()
            .map(|m| m.try_to_timed_midi_bytes(curr))
            .collect()
    }
}

macro_rules! impl_int_melody_member_to_vec_timed_midi_bytes {
    (for $($t:ty)*) => ($(
        impl ToVecTimedMidiBytes for MelodyMember<$t> {
            fn try_to_vec_timed_midi_bytes(
                &self,
                curr: u32,
            ) -> Result<Vec<(u32, Vec<u8>)>> {
                let start = self.timing.start_tick(curr)?;
                let end = self.timing.end_tick(curr)?;

                let mut ret = self.before.try_to_vec_timed_midi_bytes(start)?;

                for p in self.values.iter() {
                    if *p <= 0 || *p > 127 {
                        // There may be a cleaner way to do this
                        return Err(anyhow!(MidiError::PitchOutOfRange(format!("{:?}", p))));
                    }

                    ret.append(&mut vec![
                        (start, vec![0x90, *p as u8, self.volume]),
                        (end, vec![0x80, *p as u8, self.volume]),
                    ]);
                }

                Ok(ret)
            }
        }
    )*)
}
impl_int_melody_member_to_vec_timed_midi_bytes!(for u8 u16 u32 u64 usize i16 i32 i64);

macro_rules! impl_float_melody_member_to_vec_timed_midi_bytes {
    (for $($t:ty)*) => ($(
        impl ToVecTimedMidiBytes for MelodyMember<$t> {
            fn try_to_vec_timed_midi_bytes(
                &self,
                curr: u32,
            ) -> Result<Vec<(u32, Vec<u8>)>> {
                let start = self.timing.start_tick(curr)?;
                let end = self.timing.end_tick(curr)?;

                let mut ret = self.before.try_to_vec_timed_midi_bytes(start)?;

                for p in self.values.iter() {
                    if *p <= 0.0 || *p > 127.0 {
                        // There may be a cleaner way to do this
                        return Err(anyhow!(MidiError::PitchOutOfRange(format!("{:?}", p))));
                    }

                    let mut f = (p.fract() * 4096.0).round() as i32;
                    if f != 0 {
                        f -= 8192;
                        ret.append(&mut vec![
                            (start, vec![0xe0, (f & 0x7f) as u8, (f >> 7 & 0x7f) as u8 ]),
                            (end - 2, vec![0xe0, 0x00, 0x40 ]),
                        ]);
                    }

                    ret.append(&mut vec![
                        (start, vec![0x90, *p as u8, self.volume]),
                        (end, vec![0x80, *p as u8, self.volume]),
                    ]);
                }

                Ok(ret)
            }
        }
    )*)
}
impl_float_melody_member_to_vec_timed_midi_bytes!(for f32 f64);

macro_rules! impl_melody_to_vec_timed_midi_bytes {
    (for $($t:ty)*) => ($(
        impl ToVecTimedMidiBytes for Melody<$t> {
            fn try_to_vec_timed_midi_bytes(&self, curr: u32) -> Result<Vec<(u32, Vec<u8>)>> {
                let mut curr = curr;
                let mut ret = self.metadata.try_to_vec_timed_midi_bytes(curr)?;

                for m in self.contents.iter() {
                    ret.append(&mut m.try_to_vec_timed_midi_bytes(curr)?);
                    curr = m.timing.next_tick(curr)?;
                }

                ret.push((self.last_tick()?, END_TRACK_EVENT.to_vec()));

                Ok(ret)
            }
        }
    )*)
}
impl_melody_to_vec_timed_midi_bytes!(for u16 u32 u64 usize i16 i32 i64 f32 f64);

macro_rules! impl_melody_to_midi_bytes {
    (for $($t:ty)*) => ($(
impl ToMidiBytes for Melody<$t> {
    fn try_to_midi_bytes(&self) -> Result<Vec<u8>> {
        let mut ret: Vec<u8> = vec![];
        let mut curr = 0;

        let mut list = self.try_to_vec_timed_midi_bytes(0)?;
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for mut val in list {
            ret.append(&mut try_number_to_variable_bytes(val.0 - curr)?);
            ret.append(&mut val.1);

            curr = val.0;
        }

        let mut chunk = TRACK_HEADER_CHUNK.to_vec();

        chunk.append(&mut try_number_to_fixed_bytes(ret.len() as u32, 4)?);
        chunk.append(&mut ret);

        Ok(chunk)
    }
}
)*)
}
impl_melody_to_midi_bytes!(for u16 u32 u64 usize i16 i32 i64 f32 f64);

macro_rules! impl_score_to_midi_bytes {
    (for $($t:ty)*) => ($(
        impl ToMidiBytes for Score<$t> {
            fn try_to_midi_bytes(&self) -> Result<Vec<u8>> {
                let mut ret: Vec<u8> = HEADER_CHUNK.to_vec();
                ret.append(&mut HEADER_LENGTH.to_vec());
                ret.append(&mut HEADER_FORMAT.to_vec());
                ret.append(&mut try_number_to_fixed_bytes(self.length() as u32, 2)?);

                // Need to fix this up
                ret.append(&mut try_number_to_fixed_bytes(self.ticks_per_quarter, 2)?);

                // Need to apply score metadata to first track (but only first track)
                let mut cts: Vec<Melody<$t>> = self.clone_contents();
                if !cts.is_empty() {
                    cts[0].metadata.contents.splice(0..0, self.metadata.contents.clone());
                }

                for m in cts {
                    ret.append(&mut m.try_to_midi_bytes()?);
                }

                Ok(ret)
            }
        }

        impl WriteMidiBytes for Score<$t> {
            fn try_to_hash(&self) -> Result<u64> {
                let mut h = DefaultHasher::new();
                self.try_to_midi_bytes()?.hash(&mut h);
                Ok(h.finish())
            }

            fn try_to_write_midi_bytes(&self, file: &str) -> Result<()> {
                let mut f = File::create(file)?;

                let bytes = self.try_to_midi_bytes()?;

                f.write_all(&bytes)?;

                Ok(())
            }
        }
    )*)
}
impl_score_to_midi_bytes!(for u16 u32 u64 usize i16 i32 i64 f32 f64);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::entities::timing::{DurationalEventTiming, EventTiming};
    use crate::midi::writer::{ToMidiBytes, ToTimedMidiBytes};

    #[test]
    fn test_try_number_to_fixed_bytes() {
        assert!(try_number_to_fixed_bytes(0x00, 0).is_err());
        assert!(try_number_to_fixed_bytes(0x0100, 1).is_err());
        assert!(try_number_to_fixed_bytes(0x010000, 2).is_err());
        assert!(try_number_to_fixed_bytes(0x01000000, 3).is_err());
        assert!(try_number_to_fixed_bytes(0x00, 5).is_err());
        assert_eq!(try_number_to_fixed_bytes(0x00, 1).unwrap(), vec![0x00]);
        assert_eq!(try_number_to_fixed_bytes(0xff, 1).unwrap(), vec![0xff]);
        assert_eq!(
            try_number_to_fixed_bytes(0x0100, 2).unwrap(),
            vec![0x01, 0x00]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffff, 2).unwrap(),
            vec![0xff, 0xff]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffff, 2).unwrap(),
            vec![0xff, 0xff]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0x010000, 3).unwrap(),
            vec![0x01, 0x00, 0x00]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffffff, 3).unwrap(),
            vec![0xff, 0xff, 0xff]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0x01000000, 4).unwrap(),
            vec![0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffffffff, 4).unwrap(),
            vec![0xff, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            try_number_to_fixed_bytes(0x03f62ab3, 4).unwrap(),
            vec![0x03, 0xf6, 0x2a, 0xb3]
        );
    }

    #[test]
    fn test_try_number_to_variable_bytes() {
        assert!(try_number_to_variable_bytes(0x10000000).is_err());
        assert_eq!(try_number_to_variable_bytes(0x00).unwrap(), vec![0x00]);
        assert_eq!(try_number_to_variable_bytes(0x7f).unwrap(), vec![0x7f]);
        assert_eq!(
            try_number_to_variable_bytes(0x80).unwrap(),
            vec![0x81, 0x00]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x3fff).unwrap(),
            vec![0xff, 0x7f]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x4000).unwrap(),
            vec![0x81, 0x80, 0x00]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x1fffff).unwrap(),
            vec![0xff, 0xff, 0x7f]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x200000).unwrap(),
            vec![0x81, 0x80, 0x80, 0x00]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x0fffffff).unwrap(),
            vec![0xff, 0xff, 0xff, 0x7f]
        );
        assert_eq!(
            try_number_to_variable_bytes(0x03f62ab3).unwrap(),
            vec![0x9f, 0xd8, 0xd5, 0x33]
        );
    }

    #[test]
    fn test_try_string_to_variable_bytes() {
        assert_eq!(try_string_to_variable_bytes("").unwrap(), vec![0x00]);
        assert_eq!(
            try_string_to_variable_bytes("Per Nørgård").unwrap(),
            vec![
                0x0d, 0x50, 0x65, 0x72, 0x20, 0x4e, 0xc3, 0xb8, 0x72, 0x67, 0xc3, 0xa5, 0x72, 0x64
            ]
        );
    }

    #[test]
    fn try_to_midi_bytes() {
        assert_eq!(
            MetadataData::EndTrack.try_to_midi_bytes().unwrap(),
            vec![0xff, 0x2f, 0x00],
        );

        assert!(MetadataData::Tempo(0.0).try_to_midi_bytes().is_err());

        assert_eq!(
            MetadataData::Tempo(144.0).try_to_midi_bytes().unwrap(),
            vec![0xff, 0x51, 0x03, 0x06, 0x5b, 0x9b],
        );

        assert_eq!(
            MetadataData::Tempo(60.0).try_to_midi_bytes().unwrap(),
            vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40],
        );

        assert_eq!(
            MetadataData::Sustain(true).try_to_midi_bytes().unwrap(),
            vec![0xb0, 0x40, 0x7f],
        );

        assert_eq!(
            MetadataData::Sustain(false).try_to_midi_bytes().unwrap(),
            vec![0xb0, 0x40, 0x00],
        );

        macro_rules! test_text {
            ($k:ident, $byte:expr) => {
                assert_eq!(
                    MetadataData::$k("test".to_string())
                        .try_to_midi_bytes()
                        .unwrap(),
                    vec![0xff, $byte, 0x04, 0x74, 0x65, 0x73, 0x74]
                );
            };
        }

        test_text!(Text, 0x01);
        test_text!(Copyright, 0x02);
        test_text!(TrackName, 0x03);
        test_text!(Instrument, 0x04);
        test_text!(Lyric, 0x05);
        test_text!(Marker, 0x06);
        test_text!(CuePoint, 0x07);

        assert!(MetadataData::KeySignature("H".to_string())
            .try_to_midi_bytes()
            .is_err());

        macro_rules! test_key_sig {
            ($k:expr, $bytes: expr) => {
                assert_eq!(
                    MetadataData::KeySignature($k.to_string())
                        .try_to_midi_bytes()
                        .unwrap(),
                    $bytes.to_vec()
                )
            };
        }

        test_key_sig!("C", [0xff, 0x59, 0x02, 0x00, 0x00]);
        test_key_sig!("G", [0xff, 0x59, 0x02, 0x01, 0x00]);
        test_key_sig!("D", [0xff, 0x59, 0x02, 0x02, 0x00]);
        test_key_sig!("A", [0xff, 0x59, 0x02, 0x03, 0x00]);
        test_key_sig!("E", [0xff, 0x59, 0x02, 0x04, 0x00]);
        test_key_sig!("B", [0xff, 0x59, 0x02, 0x05, 0x00]);
        test_key_sig!("Cb", [0xff, 0x59, 0x02, 0xf9, 0x00]);
        test_key_sig!("F#", [0xff, 0x59, 0x02, 0x06, 0x00]);
        test_key_sig!("Gb", [0xff, 0x59, 0x02, 0xfa, 0x00]);
        test_key_sig!("C#", [0xff, 0x59, 0x02, 0x07, 0x00]);
        test_key_sig!("Db", [0xff, 0x59, 0x02, 0xfb, 0x00]);
        test_key_sig!("Ab", [0xff, 0x59, 0x02, 0xfc, 0x00]);
        test_key_sig!("Eb", [0xff, 0x59, 0x02, 0xfd, 0x00]);
        test_key_sig!("Bb", [0xff, 0x59, 0x02, 0xfe, 0x00]);
        test_key_sig!("F", [0xff, 0x59, 0x02, 0xff, 0x00]);
        test_key_sig!("c", [0xff, 0x59, 0x02, 0xfd, 0x01]);
        test_key_sig!("g", [0xff, 0x59, 0x02, 0xfe, 0x01]);
        test_key_sig!("d", [0xff, 0x59, 0x02, 0xff, 0x01]);
        test_key_sig!("a", [0xff, 0x59, 0x02, 0x00, 0x01]);
        test_key_sig!("e", [0xff, 0x59, 0x02, 0x01, 0x01]);
        test_key_sig!("b", [0xff, 0x59, 0x02, 0x02, 0x01]);
        test_key_sig!("f#", [0xff, 0x59, 0x02, 0x03, 0x01]);
        test_key_sig!("c#", [0xff, 0x59, 0x02, 0x04, 0x01]);
        test_key_sig!("g#", [0xff, 0x59, 0x02, 0x05, 0x01]);
        test_key_sig!("ab", [0xff, 0x59, 0x02, 0xf9, 0x01]);
        test_key_sig!("d#", [0xff, 0x59, 0x02, 0x06, 0x01]);
        test_key_sig!("eb", [0xff, 0x59, 0x02, 0xfa, 0x01]);
        test_key_sig!("a#", [0xff, 0x59, 0x02, 0x07, 0x01]);
        test_key_sig!("bb", [0xff, 0x59, 0x02, 0xfb, 0x01]);
        test_key_sig!("f", [0xff, 0x59, 0x02, 0xfc, 0x01]);

        assert!(MetadataData::TimeSignature((0, 8))
            .try_to_midi_bytes()
            .is_err());

        assert!(MetadataData::TimeSignature((4, 6))
            .try_to_midi_bytes()
            .is_err());

        assert_eq!(
            MetadataData::TimeSignature((2, 4))
                .try_to_midi_bytes()
                .unwrap(),
            vec![0xff, 0x58, 0x04, 0x02, 0x02, 0x18, 0x08]
        );

        assert_eq!(
            MetadataData::TimeSignature((9, 16))
                .try_to_midi_bytes()
                .unwrap(),
            vec![0xff, 0x58, 0x04, 0x09, 0x04, 0x18, 0x08]
        );

        assert!(MetadataData::PitchBend(2.0).try_to_midi_bytes().is_err());
        assert!(MetadataData::PitchBend(-2.0002)
            .try_to_midi_bytes()
            .is_err());

        assert_eq!(
            MetadataData::PitchBend(0.0).try_to_midi_bytes().unwrap(),
            vec![0xe0, 0x00, 0x40]
        );

        assert_eq!(
            MetadataData::PitchBend(1.9997).try_to_midi_bytes().unwrap(),
            vec![0xe0, 0x7f, 0x7f]
        );

        assert_eq!(
            MetadataData::PitchBend(-2.0).try_to_midi_bytes().unwrap(),
            vec![0xe0, 0x00, 0x00]
        );

        assert_eq!(
            MetadataData::Pan(64).try_to_midi_bytes().unwrap(),
            vec![0xb0, 0x0a, 0x40]
        );

        assert_eq!(
            MetadataData::Balance(64).try_to_midi_bytes().unwrap(),
            vec![0xb0, 0x08, 0x40]
        );

        assert_eq!(
            MetadataData::Volume(64).try_to_midi_bytes().unwrap(),
            vec![0xb0, 0x07, 0x40]
        );
    }

    #[test]
    // For this method, only test a sample as the methods it relies upon have been
    // tested more thoroughly elsewhere
    fn try_to_timed_midi_bytes() {
        assert_eq!(
            Metadata {
                data: MetadataData::EndTrack,
                timing: EventTiming {
                    tick: None,
                    offset: 32
                }
            }
            .try_to_timed_midi_bytes(64)
            .unwrap(),
            (96, vec![0xff, 0x2f, 0x00])
        );

        assert_eq!(
            Metadata {
                data: MetadataData::Tempo(60.0),
                timing: EventTiming {
                    tick: Some(128),
                    offset: 32
                }
            }
            .try_to_timed_midi_bytes(64)
            .unwrap(),
            (160, vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40])
        );
    }

    #[test]
    fn try_to_vec_timed_midi_bytes() {
        assert_eq!(
            MetadataList::default()
                .try_to_vec_timed_midi_bytes(0)
                .unwrap(),
            vec![]
        );

        assert_eq!(
            MetadataList::new(vec![
                Metadata {
                    data: MetadataData::Tempo(60.0),
                    timing: EventTiming {
                        tick: Some(128),
                        offset: 32
                    }
                },
                Metadata {
                    data: MetadataData::Text("test".to_string()),
                    timing: EventTiming {
                        tick: None,
                        offset: 64,
                    }
                }
            ])
            .try_to_vec_timed_midi_bytes(16)
            .unwrap(),
            vec![
                (160, vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40]),
                (80, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74])
            ]
        );

        macro_rules! metadata {
            ($data:expr) => {
                Metadata {
                    data: $data,
                    timing: EventTiming {
                        tick: None,
                        offset: 0,
                    },
                }
            };

            ($data:expr, $offset:expr) => {
                Metadata {
                    data: $data,
                    timing: EventTiming {
                        tick: None,
                        offset: $offset,
                    },
                }
            };

            ($data:expr, $tick:expr, $offset:expr) => {
                Metadata {
                    data: $data,
                    timing: EventTiming {
                        tick: Some($tick),
                        offset: $offset,
                    },
                }
            };
        }

        assert_eq!(
            MelodyMember {
                values: vec![60.0, 63.5, 67.0],
                timing: DurationalEventTiming {
                    tick: None,
                    offset: 0,
                    duration: 32,
                },
                volume: 64,
                before: MetadataList::new(vec![
                    metadata!(MetadataData::Sustain(true), 32, 0),
                    metadata!(MetadataData::Text("test".to_string()), 48),
                ])
            }
            .try_to_vec_timed_midi_bytes(0)
            .unwrap(),
            vec![
                (32, vec![0xb0, 0x40, 0x7f]),
                (48, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74]),
                (0, vec![0x90, 0x3c, 0x40]),
                (32, vec![0x80, 0x3c, 0x40]),
                (0, vec![0xe0, 0x0, 0x50]),
                (30, vec![0xe0, 0x0, 0x40]),
                (0, vec![0x90, 0x3f, 0x40]),
                (32, vec![0x80, 0x3f, 0x40]),
                (0, vec![0x90, 0x43, 0x40]),
                (32, vec![0x80, 0x43, 0x40]),
            ]
        );

        assert_eq!(
            MelodyMember {
                values: vec![60.0, 63.5, 67.0],
                timing: DurationalEventTiming {
                    tick: None,
                    offset: 0,
                    duration: 32,
                },
                volume: 64,
                before: MetadataList::new(vec![
                    metadata!(MetadataData::Sustain(true), 32, 0),
                    metadata!(MetadataData::Text("test".to_string()), 48),
                ])
            }
            .try_to_vec_timed_midi_bytes(160)
            .unwrap(),
            vec![
                (32, vec![0xb0, 0x40, 0x7f]),
                (208, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74]),
                (160, vec![0x90, 0x3c, 0x40]),
                (192, vec![0x80, 0x3c, 0x40]),
                (160, vec![0xe0, 0x0, 0x50]),
                (190, vec![0xe0, 0x0, 0x40]),
                (160, vec![0x90, 0x3f, 0x40]),
                (192, vec![0x80, 0x3f, 0x40]),
                (160, vec![0x90, 0x43, 0x40]),
                (192, vec![0x80, 0x43, 0x40]),
            ]
        );

        assert_eq!(
            MelodyMember {
                values: vec![60, 67],
                timing: DurationalEventTiming {
                    tick: None,
                    offset: 72,
                    duration: 32,
                },
                volume: 64,
                before: MetadataList::new(vec![
                    metadata!(MetadataData::Sustain(true), 32, 0),
                    metadata!(MetadataData::Text("test".to_string()), 48),
                ])
            }
            .try_to_vec_timed_midi_bytes(160)
            .unwrap(),
            vec![
                (32, vec![0xb0, 0x40, 0x7f]),
                (280, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74]),
                (232, vec![0x90, 0x3c, 0x40]),
                (264, vec![0x80, 0x3c, 0x40]),
                (232, vec![0x90, 0x43, 0x40]),
                (264, vec![0x80, 0x43, 0x40]),
            ]
        );

        assert_eq!(
            MelodyMember {
                values: vec![60, 67],
                timing: DurationalEventTiming {
                    tick: Some(100),
                    offset: 72,
                    duration: 32,
                },
                volume: 64,
                before: MetadataList::new(vec![
                    metadata!(MetadataData::Sustain(true), 32, 0),
                    metadata!(MetadataData::Text("test".to_string()), 48),
                ])
            }
            .try_to_vec_timed_midi_bytes(160)
            .unwrap(),
            vec![
                (32, vec![0xb0, 0x40, 0x7f]),
                (220, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74]),
                (172, vec![0x90, 0x3c, 0x40]),
                (204, vec![0x80, 0x3c, 0x40]),
                (172, vec![0x90, 0x43, 0x40]),
                (204, vec![0x80, 0x43, 0x40]),
            ]
        );

        assert_eq!(
            Melody::try_from(vec![66; 0])
                .unwrap()
                .try_to_vec_timed_midi_bytes(0)
                .unwrap(),
            vec![(0, vec![0xff, 0x2f, 0x00])],
        );

        assert_eq!(
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![60, 67],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 72,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![61],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![62],
                        timing: DurationalEventTiming {
                            tick: Some(60),
                            offset: 16,
                            duration: 64,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![63],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                ],
                metadata: MetadataList::default(),
            }
            .try_to_vec_timed_midi_bytes(0)
            .unwrap(),
            vec![
                (72, vec![0x90, 0x3c, 0x40]),
                (104, vec![0x80, 0x3c, 0x40]),
                (72, vec![0x90, 0x43, 0x40]),
                (104, vec![0x80, 0x43, 0x40]),
                (32, vec![0x90, 0x3d, 0x40]),
                (64, vec![0x80, 0x3d, 0x40]),
                (76, vec![0x90, 0x3e, 0x40]),
                (140, vec![0x80, 0x3e, 0x40]),
                (124, vec![0x90, 0x3f, 0x40]),
                (156, vec![0x80, 0x3f, 0x40]),
                (156, vec![0xff, 0x2f, 0x00]),
            ]
        );

        assert_eq!(
            Melody::<i32> {
                contents: vec![],
                metadata: MetadataList::default()
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x04, // length chunk
                0x00, 0xff, 0x2f, 0x00 // end track
            ]
        );

        assert_eq!(
            Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![60, 67],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 72,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![61],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![62],
                        timing: DurationalEventTiming {
                            tick: Some(60),
                            offset: 16,
                            duration: 64,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![63],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 32,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                ],
                metadata: MetadataList::default(),
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x2c, // length chunk
                0x20, 0x90, 0x3d, 0x40, // first note on
                0x20, 0x80, 0x3d, 0x40, // first note off
                0x08, 0x90, 0x3c, 0x40, // second note on
                0x00, 0x90, 0x43, 0x40, // third note on
                0x04, 0x90, 0x3e, 0x40, // fourth note on
                0x1c, 0x80, 0x3c, 0x40, // second note off
                0x00, 0x80, 0x43, 0x40, // third note off
                0x14, 0x90, 0x3f, 0x40, // last note on
                0x10, 0x80, 0x3e, 0x40, // fourth note off
                0x10, 0x80, 0x3f, 0x40, // last note off
                0x00, 0xff, 0x2f, 0x00, // end track
            ]
        );

        assert_eq!(
            Score {
                contents: vec![Melody::<i32> {
                    contents: vec![],
                    metadata: MetadataList::default()
                }],
                metadata: MetadataList::default(),
                ticks_per_quarter: 192,
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x68, 0x64, // file header chunk
                0x00, 0x00, 0x00, 0x06, // header length
                0x00, 0x01, // midi version
                0x00, 0x01, // number of tracks
                0x00, 0xc0, // ticks per quarter note
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x04, // track length
                0x00, 0xff, 0x2f, 0x00 // end track
            ]
        );

        assert_eq!(
            Score {
                contents: vec![Melody {
                    contents: vec![
                        MelodyMember {
                            values: vec![60, 67],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 72,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![61],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![62],
                            timing: DurationalEventTiming {
                                tick: Some(60),
                                offset: 16,
                                duration: 64,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![63],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                    ],
                    metadata: MetadataList::default(),
                }],
                metadata: MetadataList::default(),
                ticks_per_quarter: 192,
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x68, 0x64, // file header chunk
                0x00, 0x00, 0x00, 0x06, // header length
                0x00, 0x01, // midi version
                0x00, 0x01, // number of tracks
                0x00, 0xc0, // ticks per quarter note
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x2c, // track length
                0x20, 0x90, 0x3d, 0x40, // first note on
                0x20, 0x80, 0x3d, 0x40, // first note off
                0x08, 0x90, 0x3c, 0x40, // second note on
                0x00, 0x90, 0x43, 0x40, // third note on
                0x04, 0x90, 0x3e, 0x40, // fourth note on
                0x1c, 0x80, 0x3c, 0x40, // second note off
                0x00, 0x80, 0x43, 0x40, // third note off
                0x14, 0x90, 0x3f, 0x40, // last note on
                0x10, 0x80, 0x3e, 0x40, // fourth note off
                0x10, 0x80, 0x3f, 0x40, // last note off
                0x00, 0xff, 0x2f, 0x00, // end track
            ]
        );

        assert_eq!(
            Score {
                contents: vec![Melody {
                    contents: vec![
                        MelodyMember {
                            values: vec![60, 67],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 72,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![61],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![62],
                            timing: DurationalEventTiming {
                                tick: Some(60),
                                offset: 16,
                                duration: 64,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![63],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                    ],
                    metadata: MetadataList {
                        contents: vec![Metadata {
                            data: MetadataData::Tempo(144.0),
                            timing: EventTiming::default()
                        }]
                    },
                }],
                metadata: MetadataList::default(),
                ticks_per_quarter: 192,
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x68, 0x64, // file header chunk
                0x00, 0x00, 0x00, 0x06, // header length
                0x00, 0x01, // midi version
                0x00, 0x01, // number of tracks
                0x00, 0xc0, // ticks per quarter note
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x33, // track length
                0x00, 0xff, 0x51, 0x03, 0x06, 0x5b, 0x9b, // tempo of 144
                0x20, 0x90, 0x3d, 0x40, // first note on
                0x20, 0x80, 0x3d, 0x40, // first note off
                0x08, 0x90, 0x3c, 0x40, // second note on
                0x00, 0x90, 0x43, 0x40, // third note on
                0x04, 0x90, 0x3e, 0x40, // fourth note on
                0x1c, 0x80, 0x3c, 0x40, // second note off
                0x00, 0x80, 0x43, 0x40, // third note off
                0x14, 0x90, 0x3f, 0x40, // last note on
                0x10, 0x80, 0x3e, 0x40, // fourth note off
                0x10, 0x80, 0x3f, 0x40, // last note off
                0x00, 0xff, 0x2f, 0x00, // end track
            ]
        );

        assert_eq!(
            Score {
                contents: vec![Melody {
                    contents: vec![
                        MelodyMember {
                            values: vec![60, 67],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 72,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![61],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![62],
                            timing: DurationalEventTiming {
                                tick: Some(60),
                                offset: 16,
                                duration: 64,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![63],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 32,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                    ],
                    metadata: MetadataList::default(),
                }],
                metadata: MetadataList {
                    contents: vec![Metadata {
                        data: MetadataData::Tempo(144.0),
                        timing: EventTiming::default()
                    }]
                },
                ticks_per_quarter: 192,
            }
            .try_to_midi_bytes()
            .unwrap(),
            vec![
                0x4d, 0x54, 0x68, 0x64, // file header chunk
                0x00, 0x00, 0x00, 0x06, // header length
                0x00, 0x01, // midi version
                0x00, 0x01, // number of tracks
                0x00, 0xc0, // ticks per quarter note
                0x4d, 0x54, 0x72, 0x6b, // track header chunk
                0x00, 0x00, 0x00, 0x33, // track length
                0x00, 0xff, 0x51, 0x03, 0x06, 0x5b, 0x9b, // tempo of 144
                0x20, 0x90, 0x3d, 0x40, // first note on
                0x20, 0x80, 0x3d, 0x40, // first note off
                0x08, 0x90, 0x3c, 0x40, // second note on
                0x00, 0x90, 0x43, 0x40, // third note on
                0x04, 0x90, 0x3e, 0x40, // fourth note on
                0x1c, 0x80, 0x3c, 0x40, // second note off
                0x00, 0x80, 0x43, 0x40, // third note off
                0x14, 0x90, 0x3f, 0x40, // last note on
                0x10, 0x80, 0x3e, 0x40, // fourth note off
                0x10, 0x80, 0x3f, 0x40, // last note off
                0x00, 0xff, 0x2f, 0x00, // end track
            ]
        );
    }

    #[test]
    fn try_to_hash() {
        assert_eq!(
            Score {
                contents: vec![Melody {
                    contents: vec![
                        MelodyMember {
                            values: vec![60, 67],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 72,
                                duration: 96,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![61],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 96,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![62],
                            timing: DurationalEventTiming {
                                tick: Some(60),
                                offset: 16,
                                duration: 192,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                        MelodyMember {
                            values: vec![63],
                            timing: DurationalEventTiming {
                                tick: None,
                                offset: 0,
                                duration: 96,
                            },
                            volume: 64,
                            before: MetadataList::default(),
                        },
                    ],
                    metadata: MetadataList::default(),
                }],
                metadata: MetadataList::default(),
                ticks_per_quarter: 192,
            }
            .try_to_hash()
            .unwrap(),
            12932865703826558789
        );
    }

    #[test]
    fn try_to_write_midi_bytes() {
        assert!(Score {
            contents: vec![Melody {
                contents: vec![
                    MelodyMember {
                        values: vec![60, 67],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 72,
                            duration: 96,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![61],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 96,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![62],
                        timing: DurationalEventTiming {
                            tick: Some(60),
                            offset: 16,
                            duration: 192,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                    MelodyMember {
                        values: vec![63],
                        timing: DurationalEventTiming {
                            tick: None,
                            offset: 0,
                            duration: 96,
                        },
                        volume: 64,
                        before: MetadataList::default(),
                    },
                ],
                metadata: MetadataList::default(),
            }],
            metadata: MetadataList::default(),
            ticks_per_quarter: 192,
        }
        .try_to_write_midi_bytes("foo.midi")
        .is_ok());
    }
}
