use anyhow::{anyhow, Result};

pub fn from_str(sig: &str) -> Result<(u8, u16)> {
    let p: Vec<&str> = sig.split("/").collect();

    if p.len() != 2 {
        return Err(anyhow!("invalid time signature: {}", sig));
    }

    let num: u8 = p[0]
        .parse()
        .map_err(|_| anyhow!("invalid numerator in time signature: {}", sig))?;

    if num == 0 {
        return Err(anyhow!("invalid numerator in time signature: {}", sig));
    }

    let denom: u16 = p[1]
        .parse()
        .map_err(|_| anyhow!("invalid denominator in time signature: {}", sig))?;

    if denom > 0 && denom & (denom - 1) == 0 {
        Ok((num, denom))
    } else {
        Err(anyhow!("invalid denominator in time signature: {}", sig))
    }
}

pub fn to_midi_bytes(sig: &str) -> Result<[u8; 7]> {
    let bytes = from_str(sig)?;
    let byte2 = 15 - bytes.1.leading_zeros() as u8;

    Ok([0xff, 0x58, 0x04, bytes.0, byte2, 0x18, 0x08])
}

pub fn from_midi_bytes(bytes: [u8; 4]) -> Result<String, String> {
    Ok(format!("{}/{}", bytes[0], 1_u16 << bytes[1]))
}

pub fn is_valid(sig: &str) -> bool {
    from_str(sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() {
        assert!(from_str("3/8/4").is_err());
        assert!(from_str("3").is_err());
        assert!(from_str("0/4").is_err());
        assert!(from_str("f/4").is_err());
        assert!(from_str("4/f").is_err());
        assert!(from_str("256/16").is_err());
        assert!(from_str("8/7").is_err());
    }

    #[test]
    fn test_is_valid() {
        assert!(is_valid("4/4"));
        assert!(!is_valid("0/4"));
    }

    macro_rules! midi_conv_test {
        ($sig:expr, $byte1:expr, $byte2:expr) => {
            assert_eq!(
                to_midi_bytes($sig).unwrap(),
                [0xff, 0x58, 0x04, $byte1, $byte2, 0x18, 0x08]
            );
            assert_eq!(
                from_midi_bytes([$byte1, $byte2, 0x18, 0x08]),
                Ok($sig.to_string())
            );
        };
    }

    #[test]
    fn test_to_midi() {
        assert!(to_midi_bytes("3/9").is_err());

        midi_conv_test!("1/1", 0x01, 0x00);
        midi_conv_test!("4/4", 0x04, 0x02);
        midi_conv_test!("31/16", 0x1f, 0x04);
    }
}
