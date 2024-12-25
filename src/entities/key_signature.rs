// Functions for converting key signatures to and from MIDI bytes
// and for validating that a key signature is valid
const MAJOR_KEYS: [&str; 15] = [
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
];
const MINOR_KEYS: [&str; 15] = [
    "ab", "eb", "bb", "f", "c", "g", "d", "a", "e", "b", "f#", "c#", "g#", "d#", "a#",
];

fn to_midi_byte1(key: &str) -> Option<u8> {
    match key {
        "C" => Some(0),
        "a" => Some(0),
        "G" => Some(1),
        "e" => Some(1),
        "D" => Some(2),
        "b" => Some(2),
        "A" => Some(3),
        "f#" => Some(3),
        "E" => Some(4),
        "c#" => Some(4),
        "B" => Some(5),
        "g#" => Some(5),
        "F#" => Some(6),
        "d#" => Some(6),
        "C#" => Some(7),
        "a#" => Some(7),
        "Cb" => Some(249),
        "ab" => Some(249),
        "Gb" => Some(250),
        "eb" => Some(250),
        "Db" => Some(251),
        "bb" => Some(251),
        "Ab" => Some(252),
        "f" => Some(252),
        "Eb" => Some(253),
        "c" => Some(253),
        "Bb" => Some(254),
        "g" => Some(254),
        "F" => Some(255),
        "d" => Some(255),
        _ => None,
    }
}

fn to_midi_byte2(key: &str) -> u8 {
    if let Some(first) = key.chars().next() {
        if first.is_lowercase() {
            return 0x01;
        }
    }
    0x00
}

pub fn to_midi_bytes(key: &str) -> Result<[u8; 5], String> {
    if let Some(byte1) = to_midi_byte1(key) {
        let byte2 = to_midi_byte2(key);

        return Ok([0xff, 0x59, 0x02, byte1, byte2]);
    }

    Err(format!("Invalid key: {:?}", key))
}

pub fn from_midi_bytes((byte, minor): (u8, u8)) -> Result<String, String> {
    if byte > 0x07 && byte < 0xf9 {
        return Err(format!("invalid byte 1: {}", byte));
    }

    if minor > 1 {}

    let ix = (7 + byte as usize) & 0xff;

    match minor {
        0 => Ok(MAJOR_KEYS[ix].to_string()),
        1 => Ok(MINOR_KEYS[ix].to_string()),
        _ => Err(format!("invalid byte 2: {}", minor)),
    }
}

pub fn is_valid(key: &str) -> bool {
    to_midi_byte1(key).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid() {
        assert!(!is_valid(""));
        assert!(is_valid("C"));
        assert!(!is_valid("H"));
        assert!(is_valid("Gb"));
    }

    macro_rules! midi_conv_test {
        ($key:expr, $byte1:expr, $byte2:expr) => {
            assert_eq!(to_midi_bytes($key), Ok([0xff, 0x59, 0x02, $byte1, $byte2]));
            assert_eq!(from_midi_bytes(($byte1, $byte2)), Ok($key.to_string()));
        };
    }

    #[test]
    fn midi_bytes_conversions() {
        assert!(to_midi_bytes("Dc").is_err());
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
