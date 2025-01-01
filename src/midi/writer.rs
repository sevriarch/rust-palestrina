use crate::entities::timing::Timing;
use crate::metadata::data::{Metadata, MetadataData};

pub trait ToTimedMidiBytes {
    fn try_to_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>), String>;
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

fn try_metadata_to_midi_bytes(d: &MetadataData) -> Result<Vec<u8>, String> {
    match d {
        MetadataData::EndTrack => Ok(vec![0xff, 0x2f, 0x00]),
        MetadataData::Tempo(f32) => {
            let mut ret = vec![0xff, 0x51, 0x03];
            ret.append(&mut try_number_to_fixed_bytes(
                (6e7 / f32).round() as u32,
                3,
            )?);

            Ok(ret)
        }
        MetadataData::Text(txt) => {
            let mut ret = vec![0xff, 0x01];
            ret.append(&mut try_string_to_variable_bytes(txt)?);

            Ok(ret)
        }
        /*
        MetadataData::Sustain(bool) => Ok(vec![ ])
        MetadataData::KeySignature(String),
        MetadataData::TimeSignature(String),
        MetadataData::Instrument(String),
        MetadataData::Lyric(String),
        MetadataData::Marker(String),
        MetadataData::CuePoint(String),
        MetadataData::Copyright(String),
        MetadataData::TrackName(String),
        MetadataData::Volume(i16),
        MetadataData::Pan(i16),
        MetadataData::Balance(i16),
        MetadataData::PitchBend(i16),
        */
        _ => Err("invalid metadata".to_string()),
    }
}

impl ToTimedMidiBytes for Metadata {
    fn try_to_midi_bytes(&self, curr: u32) -> Result<(u32, Vec<u8>), String> {
        let databytes = try_metadata_to_midi_bytes(&self.data)?;

        Ok((self.timing.start_tick(curr)?, databytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_try_metadata_to_midi_bytes() {
        assert_eq!(
            try_metadata_to_midi_bytes(&MetadataData::EndTrack),
            Ok(vec![0xff, 0x2f, 0x00]),
        );

        assert_eq!(
            try_metadata_to_midi_bytes(&MetadataData::Tempo(144.0)),
            Ok(vec![0xff, 0x51, 0x03, 0x06, 0x5b, 0x9b]),
        );

        assert_eq!(
            try_metadata_to_midi_bytes(&MetadataData::Tempo(60.0)),
            Ok(vec![0xff, 0x51, 0x03, 0x0f, 0x42, 0x40]),
        );
    }
}
