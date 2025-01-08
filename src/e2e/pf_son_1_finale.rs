use crate::collections::traits::Collection;
use crate::constants::dynamics;
use crate::imports::{constant, infinity_var1, linear, sinusoidal};
use crate::midi::writer::WriteMidiBytes;
use crate::ops::pitch::Pitch;
use crate::score::Score;
use crate::sequences::{melody::Melody, note::NoteSeq, numeric::NumericSeq, traits::Sequence};
use anyhow::Result;

const LEN: usize = 1280;
const TICKS: u32 = 128;
const GAP: usize = 5;
const MOD: i32 = 25;
const BAR: i32 = 16;

fn bar(x: i32) -> i32 {
    (x - 1) * BAR
}

fn make_overlay() -> Result<NumericSeq<i32>> {
    let vals = sinusoidal(LEN, 8.5, 0.0, 3600.0);
    let overlay = NumericSeq::new(vals);

    let max = overlay.max_value().unwrap();
    let min = overlay.min_value().unwrap();

    overlay
        .mutate_slice(bar(29), bar(37), |v| v.invert_pitch(max))?
        .mutate_slice(bar(37), bar(45), |v| *v = min - *v / 2)?
        .mutate_slice(bar(61), bar(69), |v| v.invert_pitch(max))?
        .set_slice(bar(69), bar(73), min)?
        .set_slice(bar(73), bar(77), max + 8)?
        .mutate_slice_indexed(bar(77), bar(81), |(i, v)| {
            *v = min + ((i as i32 - 76 * 16 + 1) * 8 / BAR)
        })?
        .append_items(&[8, 8])
        .transpose(-16)
}

// Raw musical material using every nth note of a variant of Per Nørgård's infinity series,
// with 0 mapped to the A below middle C
fn make_melody(offset: usize) -> Result<NumericSeq<i32>> {
    NumericSeq::new(infinity_var1(LEN * GAP))
        .keep_nth_from(GAP, offset)?
        .modulus(MOD)?
        .transpose(57)
}

// Function to transpose each member of the passed sequence up or down in pitch according to
// the value of the corresponding member of the overlay sequence
fn apply_overlay(seq: NoteSeq<i32>) -> Result<NoteSeq<i32>> {
    seq.combine(
        |(a, b)| {
            if let Some(base) = a {
                Some(base + b.unwrap_or_default())
            } else {
                *a
            }
        },
        make_overlay()?.try_into()?,
    )
}

fn make_volume() -> Result<NumericSeq<u8>> {
    let mut coll = NumericSeq::new(constant(LEN, dynamics::MF as i32));

    coll = coll.append_items(&[dynamics::PP as i32, dynamics::PP as i32]);

    macro_rules! dy {
        ($from:expr, $to:expr, $const:expr) => {
            coll = coll.set_slice(bar($from), bar($to), $const as i32)?;
        };

        ($from:expr, $to:expr, $first:expr, $last:expr) => {
            coll = coll.replace_slice(
                bar($from),
                bar($to),
                linear((($to - $from) * BAR) as usize, $first as i32, $last as i32),
            )?;
        };
    }

    dy!(3, 6, dynamics::MF, dynamics::F);
    dy!(6, 8, dynamics::F);
    dy!(8, 10, dynamics::F, dynamics::MF);
    dy!(12, 14, dynamics::MF, dynamics::F);
    dy!(14, 16, dynamics::F);
    dy!(16, 18, dynamics::F, dynamics::MF);
    dy!(20, 22, dynamics::MF, dynamics::F);
    dy!(22, 24, dynamics::F);
    dy!(24, 26, dynamics::F, dynamics::MF);
    dy!(27, 29, dynamics::MF, dynamics::P);
    dy!(33, 36, dynamics::MF, dynamics::MP);
    dy!(36, 37, dynamics::MP);
    dy!(37, 42, dynamics::P, dynamics::PP);
    dy!(42, 46, dynamics::PP);
    dy!(46, 48, dynamics::P, dynamics::MP);
    dy!(48, 49, dynamics::MP);
    dy!(49, 53, dynamics::MP, dynamics::P);
    dy!(53, 56, dynamics::PP);
    dy!(56, 59, dynamics::PP, dynamics::P);
    dy!(59, 61, dynamics::P, dynamics::PP);
    dy!(61, 69, dynamics::MP);
    dy!(69, 71, dynamics::PP);
    dy!(71, 75, dynamics::PP, dynamics::MF);
    dy!(77, 80, dynamics::P, dynamics::F);
    dy!(80, 81, dynamics::F, dynamics::PP);

    Ok(NumericSeq::new(
        coll.contents.iter().map(|v| (*v % 256) as u8).collect(),
    ))
}

fn melodify(seq: NoteSeq<i32>) -> Result<Melody<i32>> {
    let mel: Melody<i32> = apply_overlay(seq)?.try_into()?;

    mel.mutate_each(|m| {
        if !m.is_silent() {
            m.values = vec![m.values[0], m.values[0] + 12];
        }
    })
    .with_duration(TICKS / 4)
    .with_volumes(make_volume()?.contents)
}

fn make_score() -> Result<()> {
    // Collect raw material using multiple variants of the base sequence
    let mut lines = Vec::with_capacity(GAP);

    for i in 0..GAP {
        lines.push(make_melody(i)?);
    }

    let first = lines.remove(0);

    // Take the different variants and for every 8 notes select the following:
    // [ variant 0, variant 0, variant 0, variant 1, variant 1, variant 1, variant 2, variant 2 ]
    //
    // For those who are observant, yes, two of the variants (3 and 4) are never used, and only
    // the first third of the `look` variable is used. This can happen when adjusting algorithms
    // till you get a result you like.
    let look: Vec<usize> = vec![
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1,
    ];

    let zig: NoteSeq<i32> = first
        .map_with_indexed(|(i, vals)| *vals[look[i % 8]], lines)?
        .append_items(&[69, 97])
        .try_into()?;

    let zag = zig.partition_in_position(
        |p| match p {
            Some(pitch) => *pitch >= 71,
            _ => false,
        },
        None,
    )?;

    let m0 = melodify(zag.0)?;
    let m1 = melodify(zag.1)?;
    println!("{:?}", m1);
    let score = Score::new(vec![m0, m1]).with_ticks_per_quarter(TICKS);

    score.try_to_write_midi_bytes("src/e2e/pf_son_1_finale.midi")
}
/*
    score([ melodify(zag[0]), melodify(zag[1]) ])
        .withTicksPerQuarter(TICKS)
        .withTempo(100)
        .withTimeSignature('4/4')
        .writeCanvas(__filename)
        .writeMidi(__filename)
        .expectHash(HASH)
*/

#[test]
fn test_e2e() {
    assert!(make_score().is_ok());
}
