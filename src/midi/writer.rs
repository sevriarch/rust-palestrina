use crate::entities::{key_signature, time_signature, timing::Timing};
use crate::metadata::{
    data::{Metadata, MetadataData},
    list::MetadataList,
};
use crate::sequences::melody::MelodyMember;

use super::constants::*;

pub trait ToMidiBytes {
    fn try_to_midi_bytes(&self) -> Result<Vec<u8>, String>;
}

pub trait ToTimedMidiBytes {
    fn try_to_timed_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>), String>;
}

pub trait ToVecTimedMidiBytes {
    fn try_to_vec_timed_midi_bytes(&self, curr: u32) -> Result<Vec<(u32, Vec<u8>)>, String>;
}

fn try_number_to_fixed_bytes(num: u32, size: usize) -> Result<Vec<u8>, String> {
    if size == 0 {
        return Err("can't convert a number to zero bytes".to_string());
    }

    let mut bytes = Vec::<u8>::with_capacity(size);
    let mut n = num;
    bytes.push((n & 0xff) as u8);

    for _ in 1..size {
        n >>= 8;
        bytes.push((n & 0xff) as u8);
    }

    if n > 0xff {
        return Err(format!("number {} cannot fit into {} bytes", n, size));
    }

    bytes.reverse();
    Ok(bytes)
}

fn try_number_to_variable_bytes(num: u32) -> Result<Vec<u8>, String> {
    if num > 0xfffffff {
        return Err(format!("number {:?} >= 2^28", num));
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

fn try_string_to_variable_bytes(str: &str) -> Result<Vec<u8>, String> {
    let mut chars: Vec<u8> = str
        .chars()
        .flat_map(|c| c.to_string().into_bytes())
        .collect();
    let mut ret = try_number_to_variable_bytes(chars.len() as u32)?;

    ret.append(&mut chars);

    Ok(ret)
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
    fn try_to_midi_bytes(&self) -> Result<Vec<u8>, String> {
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
            MetadataData::Sustain(val) => Ok(vec![
                CONTROLLER_BYTE,
                SUSTAIN_CONTROLLER,
                if *val {
                    EVENT_ON_VALUE
                } else {
                    EVENT_OFF_VALUE
                },
            ]),
            MetadataData::KeySignature(val) => Ok(key_signature::to_midi_bytes(val)?.to_vec()),
            MetadataData::TimeSignature(val) => Ok(time_signature::to_midi_bytes(val)?.to_vec()),
            /*
            MetadataData::Instrument(String),
            MetadataData::Volume(i16),
            MetadataData::Pan(i16),
            MetadataData::Balance(i16),
            MetadataData::PitchBend(i16),
            */
            _ => Err("invalid metadata".to_string()),
        }
    }
}

impl ToTimedMidiBytes for Metadata {
    fn try_to_timed_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>), String> {
        let databytes = self.data.try_to_midi_bytes()?;

        Ok((self.timing.start_tick(curr)?, databytes))
    }
}

impl ToVecTimedMidiBytes for MetadataList {
    fn try_to_vec_timed_midi_bytes(&self, curr: u32) -> Result<Vec<(u32, Vec<u8>)>, String> {
        self.contents
            .iter()
            .map(|m| m.try_to_timed_midi_bytes(curr))
            .collect()
    }
}

macro_rules! impl_melody_member_to_vec_timed_midi_bytes {
    (for $($t:ty)*) => ($(
        impl ToVecTimedMidiBytes for MelodyMember<$t> {
            fn try_to_vec_timed_midi_bytes(
                &self,
                curr: u32,
            ) -> Result<Vec<(u32, Vec<u8>)>, String> {
                let start = self.timing.start_tick(curr)?;
                let end = self.timing.end_tick(curr)?;

                let mut ret = self.before.try_to_vec_timed_midi_bytes(start)?;

                for p in self.values.iter() {
                    if *p <= 0 || *p > 127 {
                        return Err("pitch out of range".to_string());
                    }

                    ret.append(&mut vec![
                        (start, vec![0x90, *p as u8]),
                        (end, vec![0x80, *p as u8]),
                    ]);
                }

                Ok(ret)
            }
        }
    )*)
}

impl_melody_member_to_vec_timed_midi_bytes!(for u8 u16 u32 u64 usize i16 i32 i64);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::entities::timing::EventTiming;
    use crate::midi::writer::{ToMidiBytes, ToTimedMidiBytes};

    #[test]
    fn test_try_number_to_fixed_bytes() {
        assert!(try_number_to_fixed_bytes(0x00, 0).is_err());
        assert!(try_number_to_fixed_bytes(0x0100, 1).is_err());
        assert!(try_number_to_fixed_bytes(0x010000, 2).is_err());
        assert!(try_number_to_fixed_bytes(0x01000000, 3).is_err());
        assert_eq!(try_number_to_fixed_bytes(0x00, 1), Ok(vec![0x00]));
        assert_eq!(try_number_to_fixed_bytes(0xff, 1), Ok(vec![0xff]));
        assert_eq!(try_number_to_fixed_bytes(0x0100, 2), Ok(vec![0x01, 0x00]));
        assert_eq!(try_number_to_fixed_bytes(0xffff, 2), Ok(vec![0xff, 0xff]));
        assert_eq!(try_number_to_fixed_bytes(0xffff, 2), Ok(vec![0xff, 0xff]));
        assert_eq!(
            try_number_to_fixed_bytes(0x010000, 3),
            Ok(vec![0x01, 0x00, 0x00])
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffffff, 3),
            Ok(vec![0xff, 0xff, 0xff])
        );
        assert_eq!(
            try_number_to_fixed_bytes(0x01000000, 4),
            Ok(vec![0x01, 0x00, 0x00, 0x00])
        );
        assert_eq!(
            try_number_to_fixed_bytes(0xffffffff, 4),
            Ok(vec![0xff, 0xff, 0xff, 0xff])
        );
        assert_eq!(
            try_number_to_fixed_bytes(0x03f62ab3, 4),
            Ok(vec![0x03, 0xf6, 0x2a, 0xb3])
        );
    }

    #[test]
    fn test_try_number_to_variable_bytes() {
        assert!(try_number_to_variable_bytes(0x10000000).is_err());
        assert_eq!(try_number_to_variable_bytes(0x00), Ok(vec![0x00]));
        assert_eq!(try_number_to_variable_bytes(0x7f), Ok(vec![0x7f]));
        assert_eq!(try_number_to_variable_bytes(0x80), Ok(vec![0x81, 0x00]));
        assert_eq!(try_number_to_variable_bytes(0x3fff), Ok(vec![0xff, 0x7f]));
        assert_eq!(
            try_number_to_variable_bytes(0x4000),
            Ok(vec![0x81, 0x80, 0x00])
        );
        assert_eq!(
            try_number_to_variable_bytes(0x1fffff),
            Ok(vec![0xff, 0xff, 0x7f])
        );
        assert_eq!(
            try_number_to_variable_bytes(0x200000),
            Ok(vec![0x81, 0x80, 0x80, 0x00])
        );
        assert_eq!(
            try_number_to_variable_bytes(0x0fffffff),
            Ok(vec![0xff, 0xff, 0xff, 0x7f])
        );
        assert_eq!(
            try_number_to_variable_bytes(0x03f62ab3),
            Ok(vec![0x9f, 0xd8, 0xd5, 0x33])
        );
    }

    #[test]
    fn test_try_string_to_variable_bytes() {
        assert_eq!(try_string_to_variable_bytes(""), Ok(vec![0x00]));
        assert_eq!(
            try_string_to_variable_bytes("Per Nørgård"),
            Ok(vec![
                0x0d, 0x50, 0x65, 0x72, 0x20, 0x4e, 0xc3, 0xb8, 0x72, 0x67, 0xc3, 0xa5, 0x72, 0x64
            ])
        );
    }

    #[test]
    fn try_to_midi_bytes() {
        assert_eq!(
            MetadataData::EndTrack.try_to_midi_bytes(),
            Ok(vec![0xff, 0x2f, 0x00]),
        );

        assert_eq!(
            MetadataData::Tempo(144.0).try_to_midi_bytes(),
            Ok(vec![0xff, 0x51, 0x03, 0x06, 0x5b, 0x9b]),
        );

        assert_eq!(
            MetadataData::Tempo(60.0).try_to_midi_bytes(),
            Ok(vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40]),
        );

        assert_eq!(
            MetadataData::Sustain(true).try_to_midi_bytes(),
            Ok(vec![0xb0, 0x40, 0x7f]),
        );

        assert_eq!(
            MetadataData::Sustain(false).try_to_midi_bytes(),
            Ok(vec![0xb0, 0x40, 0x00]),
        );

        macro_rules! test_text {
            ($k:ident, $byte:expr) => {
                assert_eq!(
                    MetadataData::$k("test".to_string()).try_to_midi_bytes(),
                    Ok(vec![0xff, $byte, 0x04, 0x74, 0x65, 0x73, 0x74])
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

        macro_rules! test_key_sig {
            ($k:expr, $bytes: expr) => {
                assert_eq!(
                    MetadataData::KeySignature($k.to_string()).try_to_midi_bytes(),
                    Ok($bytes.to_vec())
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

        assert!(MetadataData::TimeSignature("0/8".to_string())
            .try_to_midi_bytes()
            .is_err());

        assert!(MetadataData::TimeSignature("4/6".to_string())
            .try_to_midi_bytes()
            .is_err());

        assert_eq!(
            MetadataData::TimeSignature("2/4".to_string()).try_to_midi_bytes(),
            Ok(vec![0xff, 0x58, 0x04, 0x02, 0x02, 0x18, 0x08])
        );

        assert_eq!(
            MetadataData::TimeSignature("9/16".to_string()).try_to_midi_bytes(),
            Ok(vec![0xff, 0x58, 0x04, 0x09, 0x04, 0x18, 0x08])
        );
        //[ { event: 'time-signature', value: '9/16' }, undefined, [ 0xff, 0x58, 0x04, 0x09, 0x04, 0x18, 0x08 ] ],
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
            .try_to_timed_midi_bytes(64),
            Ok((96, vec![0xff, 0x2f, 0x00]))
        );

        assert_eq!(
            Metadata {
                data: MetadataData::Tempo(60.0),
                timing: EventTiming {
                    tick: Some(128),
                    offset: 32
                }
            }
            .try_to_timed_midi_bytes(64),
            Ok((160, vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40]))
        );
    }

    #[test]
    fn try_to_vec_timed_midi_bytes() {
        assert_eq!(
            MetadataList::default().try_to_vec_timed_midi_bytes(0),
            Ok(vec![])
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
            .try_to_vec_timed_midi_bytes(16),
            Ok(vec![
                (160, vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40]),
                (80, vec![0xff, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74])
            ])
        );
    }
}
