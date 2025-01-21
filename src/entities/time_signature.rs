use anyhow::Result;

pub fn from_midi_bytes(bytes: [u8; 4]) -> Result<String, String> {
    Ok(format!("{}/{}", bytes[0], 1_u16 << bytes[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! midi_conv_test {
        ($sig:expr, $byte1:expr, $byte2:expr) => {
            assert_eq!(
                from_midi_bytes([$byte1, $byte2, 0x18, 0x08]),
                Ok($sig.to_string())
            );
        };
    }

    #[test]
    fn test_from_midi() {
        midi_conv_test!("1/1", 0x01, 0x00);
        midi_conv_test!("4/4", 0x04, 0x02);
        midi_conv_test!("31/16", 0x1f, 0x04);
    }
}
