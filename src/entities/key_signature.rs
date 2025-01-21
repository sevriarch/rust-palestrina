use anyhow::{anyhow, Result};

// Functions for converting key signatures to and from MIDI bytes
// and for validating that a key signature is valid
const MAJOR_KEYS: [&str; 15] = [
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
];
const MINOR_KEYS: [&str; 15] = [
    "ab", "eb", "bb", "f", "c", "g", "d", "a", "e", "b", "f#", "c#", "g#", "d#", "a#",
];

pub fn from_midi_bytes((byte, minor): (u8, u8)) -> Result<String> {
    if byte > 0x07 && byte < 0xf9 {
        return Err(anyhow!("invalid byte 1: {}", byte));
    }

    let ix = (7 + byte as usize) & 0xff;

    match minor {
        0 => Ok(MAJOR_KEYS[ix].to_string()),
        1 => Ok(MINOR_KEYS[ix].to_string()),
        _ => Err(anyhow!("invalid byte 2: {}", minor)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! midi_conv_test {
        ($key:expr, $byte1:expr, $byte2:expr) => {
            assert_eq!(from_midi_bytes(($byte1, $byte2)).unwrap(), $key.to_string());
        };
    }

    #[test]
    fn midi_bytes_conversions() {
        assert!(from_midi_bytes((0x08, 0x00)).is_err());
        assert!(from_midi_bytes((0xf8, 0x00)).is_err());
        assert!(from_midi_bytes((0x01, 0x02)).is_err());

        midi_conv_test!("C", 0x00, 0x00);
        midi_conv_test!("G", 0x01, 0x00);
        midi_conv_test!("D", 0x02, 0x00);
        midi_conv_test!("A", 0x03, 0x00);
        midi_conv_test!("E", 0x04, 0x00);
        midi_conv_test!("B", 0x05, 0x00);
        midi_conv_test!("F#", 0x06, 0x00);
        midi_conv_test!("C#", 0x07, 0x00);
        midi_conv_test!("F", 0xff, 0x00);
        midi_conv_test!("Bb", 0xfe, 0x00);
        midi_conv_test!("Eb", 0xfd, 0x00);
        midi_conv_test!("Ab", 0xfc, 0x00);
        midi_conv_test!("Db", 0xfb, 0x00);
        midi_conv_test!("Gb", 0xfa, 0x00);
        midi_conv_test!("Cb", 0xf9, 0x00);
        midi_conv_test!("a", 0x00, 0x01);
        midi_conv_test!("e", 0x01, 0x01);
        midi_conv_test!("b", 0x02, 0x01);
        midi_conv_test!("f#", 0x03, 0x01);
        midi_conv_test!("c#", 0x04, 0x01);
        midi_conv_test!("g#", 0x05, 0x01);
        midi_conv_test!("d#", 0x06, 0x01);
        midi_conv_test!("a#", 0x07, 0x01);
        midi_conv_test!("d", 0xff, 0x01);
        midi_conv_test!("g", 0xfe, 0x01);
        midi_conv_test!("c", 0xfd, 0x01);
        midi_conv_test!("f", 0xfc, 0x01);
        midi_conv_test!("bb", 0xfb, 0x01);
        midi_conv_test!("eb", 0xfa, 0x01);
        midi_conv_test!("ab", 0xf9, 0x01);
    }
}
