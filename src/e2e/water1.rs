/*'use strict'

const { imports, noteseq, NoteSeqMember, melody, MelodyMember, score, Melody } = require('../built')

const HASH  = 'd0422047681ce6632afd75527f33c3ef'

const META = [
    { event: 'text', value: 'Triptych I: Water' },
    { event: 'copyright', value: 'C2019 sevriarch@gmail.com' },
    { event: 'marker', value: 'Section 1' },
    { event: 'marker', value: 'Section 2', at: 378 * TICKS * 2 },
    { event: 'marker', value: 'Section 3', at: 602 * TICKS * 2 },
]

const start = process.hrtime()

// Set to true if silent notes should be a C#-1 to help with Sibelius tuplet imports


const DEBUG = process.env.WATER_DEBUG && process.env.WATER_DEBUG !== '0'

*/
use crate::collections::traits::Collection;
use crate::entities::scale::Scale;
use crate::imports::rasmussen;
use crate::metadata::Metadata;
use crate::ops::pitch::Pitch;
use crate::sequences::{note::NoteSeq, numeric::NumericSeq, traits::Sequence};
use anyhow::Result;

const LEN: usize = 2816;
const TICKS: u32 = 840;

//const MAXGAP: i8 = 15;

const V_PP: u8 = 15;
//const V_P: u8 = 30;
//const V_MP: u8 = 45;
//const V_MF: u8 = 60;
//const V_F: u8 = 75;
//const V_FF: u8 = 90;
const V_FFF: u8 = 100;
const V_RANGE: u8 = V_FFF - V_PP;

//const SIB_IMPORT: bool = false;

pub fn build_base_sequences() -> Result<(NoteSeq<i32>, NoteSeq<i32>)> {
    let base: NoteSeq<i32> = NoteSeq::try_from(rasmussen(LEN * 3))?;
    let m1 = base
        .clone()
        .keep_nth(2)?
        .modulus(33)
        .clone()
        .dedupe()
        .invert_pitch(8)
        .clone()
        .replace_indices(&[0], &[None])?
        .mutate_each_enumerated(|(i, v)| {
            if i > 120 && v.is_some_and(|v| v * 110 < (-760 - i as i32)) {
                v.transpose_pitch(8);
            }
        })
        .scale(Scale::from_name("octatonic21")?, 0)?;
    let m2 = base
        .clone()
        .drop_nth(2)?
        .modulus(33)
        .clone()
        .dedupe()
        .transpose(-16)?
        .retrograde()?
        .scale(Scale::from_name("octatonic21")?, 0)?;

    Ok((m1, m2))
}

fn dynamics1(len: usize) -> NumericSeq<u8> {
    fn cresc(val: f32) -> u8 {
        V_PP + (val * V_RANGE as f32 / 2.0) as u8
    }

    fn dim(val: f32) -> u8 {
        V_FFF - (val * V_RANGE as f32 / 2.0) as u8
    }

    fn vol(ix: usize, len: usize) -> u8 {
        let mut loc = 8.0 * ix as f32 / len as f32;

        if ix < len / 16 {
            return cresc(loc * 2.0);
        }

        if ix < len / 8 {
            return dim(loc * 2.0);
        }

        loc %= 8.0;

        if loc < 2.0 {
            return V_PP;
        }
        if loc < 4.0 {
            return cresc(loc - 2.000000001);
        }
        if loc < 6.0 {
            return V_FFF;
        }

        dim(loc - 6.0)
    }

    let mut ret: Vec<u8> = (0..len).map(|v| vol(v, len)).collect();

    // TODO: figure out if there's a way to adjust algorithm to correct these
    ret[528] = 31;
    ret[792] = 83;
    ret[1496] = 67;
    ret[1584] = 49;

    NumericSeq::new(ret)
}

#[test]
fn test_dynamics() {
    let a = dynamics1(1760);
    let b = NumericSeq::new(vec![
        15, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 23, 23,
        23, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 28, 28, 28, 29, 29, 30, 30, 30, 31, 31, 32, 32,
        32, 33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 40, 40, 40, 41,
        41, 42, 42, 42, 43, 43, 43, 44, 44, 45, 45, 45, 46, 46, 47, 47, 47, 48, 48, 49, 49, 49, 50,
        50, 50, 51, 51, 52, 52, 52, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 57, 58, 58, 57, 57, 56,
        56, 56, 55, 55, 55, 54, 54, 53, 53, 53, 52, 52, 51, 51, 51, 50, 50, 49, 49, 49, 48, 48, 48,
        47, 47, 46, 46, 46, 45, 45, 44, 44, 44, 43, 43, 43, 42, 42, 41, 41, 41, 40, 40, 39, 39, 39,
        38, 38, 38, 37, 37, 36, 36, 36, 35, 35, 34, 34, 34, 33, 33, 32, 32, 32, 31, 31, 31, 30, 30,
        29, 29, 29, 28, 28, 27, 27, 27, 26, 26, 26, 25, 25, 24, 24, 24, 23, 23, 22, 22, 22, 21, 21,
        21, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18,
        18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23,
        23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27,
        27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31,
        32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36,
        36, 36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40,
        41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45,
        45, 45, 45, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49,
        49, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 54, 54,
        54, 54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58,
        58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 63,
        63, 63, 63, 63, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 67, 67, 67,
        67, 67, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71,
        72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 76, 76, 76,
        76, 76, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80,
        81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 85, 85,
        85, 85, 85, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89,
        89, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 94, 94,
        94, 94, 94, 94, 95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98,
        98, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 99, 99, 99, 99, 99, 98, 98, 98, 98, 98, 97, 97, 97, 97, 97, 96, 96, 96, 96, 96, 95,
        95, 95, 95, 95, 95, 94, 94, 94, 94, 94, 93, 93, 93, 93, 93, 92, 92, 92, 92, 92, 91, 91, 91,
        91, 91, 90, 90, 90, 90, 90, 89, 89, 89, 89, 89, 89, 88, 88, 88, 88, 88, 87, 87, 87, 87, 87,
        86, 86, 86, 86, 86, 85, 85, 85, 85, 85, 84, 84, 84, 84, 84, 83, 83, 83, 83, 83, 83, 82, 82,
        82, 82, 82, 81, 81, 81, 81, 81, 80, 80, 80, 80, 80, 79, 79, 79, 79, 79, 78, 78, 78, 78, 78,
        78, 77, 77, 77, 77, 77, 76, 76, 76, 76, 76, 75, 75, 75, 75, 75, 74, 74, 74, 74, 74, 73, 73,
        73, 73, 73, 72, 72, 72, 72, 72, 72, 71, 71, 71, 71, 71, 70, 70, 70, 70, 70, 69, 69, 69, 69,
        69, 68, 68, 68, 68, 68, 67, 67, 67, 67, 67, 67, 66, 66, 66, 66, 66, 65, 65, 65, 65, 65, 64,
        64, 64, 64, 64, 63, 63, 63, 63, 63, 62, 62, 62, 62, 62, 61, 61, 61, 61, 61, 61, 60, 60, 60,
        60, 60, 59, 59, 59, 59, 59, 58, 58, 58, 58, 58, 57, 57, 57, 57, 57, 56, 56, 56, 56, 56, 55,
        55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 51, 51, 51,
        51, 51, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 47, 47, 47, 47, 47,
        46, 46, 46, 46, 46, 45, 45, 45, 45, 45, 44, 44, 44, 44, 44, 44, 43, 43, 43, 43, 43, 42, 42,
        42, 42, 42, 41, 41, 41, 41, 41, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38,
        38, 37, 37, 37, 37, 37, 36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 33, 33,
        33, 33, 33, 33, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29, 29, 29, 29,
        29, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 24,
        24, 24, 24, 24, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 20, 20, 20,
        20, 20, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16,
    ]);

    (0..1760).for_each(|i| assert_eq!((i, a.contents[i]), (i, b.contents[i])));
}

/*
function log(...args) {
    console.log(new Date(), ...args)
}

function dbg(...args) {
    if (DEBUG) log(...args)
}

function logTrack(n, section, s) {
    const len = s.length
    const sil = s.filter(e => e.isSilent()).length

    log('track', n, section + ': len', len, 'notes', len - sil, 'silent', sil)
}

*/
pub fn rhythm(n: u32) -> u32 {
    TICKS * 2 / (8 - n)
}

pub fn tempo_events() -> Result<Vec<Metadata>> {
    vec![
        (128, 1),
        (124, 325),
        (120, 343),
        (116, 361),
        (112, 379), // "Tempo flessibile"
        (108, 462),
        (104, 542),
        (108, 570),
        (112, 581),
        (116, 592),
        (120, 603),
        (124, 614),
        (104, 622),
        (108, 626),
        (112, 630),
        (116, 633),
        (120, 636),
        (128, 639),
        (120, 656),
        (112, 660),
        (104, 664),
        (96, 668),
        (88, 672),
        (80, 676),
        (72, 680),
        (64, 684),
    ]
    .iter()
    .map(|(t, bar)| Metadata::try_from(("tempo", *t as f32, Some((bar - 1) * 2 * TICKS), 0)))
    .collect()
}
/*
fn melody1(n: usize) {
    //log('entering m1', n)

    let barlen = 8 - n;
    let len1 = (8 - n) * LEN / 8;
    let rhy1 = rhythm(n.try_into().unwrap());
    let vol1 = dynamics1(len1);

    let first = [ 30, 29, 28, 55 ][n];
    let p_gap = 28 * barlen; // 28 bars

    let mut s_on = vec![ 0 ];
    let mut s_off = vec![];

    (barlen * first..len1).step_by(p_gap).for_each(|i| {
        s_off.push(i - 2);
        s_on.push(i);
    });

    let sieve1 = |(i, v)| {
        let rotator = match v {
            2 => 5,
            3 => 1,
            5 => 3,
            6 => 0,
            8 => 4,
            9 => 2,
            11 => 6,
            _ => 0,
        };

        let m = (48 + v) % 12;
    });

    function sieve1(v, i) {
        const rotators = { 2: 5, 3: 1, 5: 3, 6: 0, 8: 4, 9: 2, 11: 6 }

        const GAP  = 128 // 16 bars
        const LPAD = 1024 // 128 bars

        const m = (48 + v) % 12
        const n = null
        const s = (i - LPAD) / GAP

        function rotato(x) {
            if (s < x) { return v }
            if (s < (13 - x)) { return n }
            if (s < (14 + x)) { return -v }
            if (s < (27 - x)) { return n }
            return v
        }

        return rotato(rotators[m])
    }

    return m1
        .keep(len1)
        .if(!n)
            .then(s => s.replaceIndices(-1, null))
        .mapPitch(sieve1)
        .transpose(84 - 12 * n)
        .toMelody()
        .withDuration(rhy1)
        .withVolume(vol1)
        .withStartTick(n * TICKS * 14)
        .withEventBefore(s_on, 'sustain', 1)
        .withEventAfter(s_off, 'sustain', 0)
        .tap(s => logTrack(n, 'part1', s))
}
*/

/*
function melody1(n) {
    log('entering m1', n)

    const BARLEN = 8 - n

    const len1 = (8 - n) * LEN / 8

    const rhy1 = rhythm(n)
    const vol1 = dynamics1(len1)

    function pedal1(n) {
        const FIRST = [ 30, 29, 28, 55 ][n]
        const P_GAP = 28 * BARLEN // 28 bars

        const s_on = [ 0 ]
        const s_off = []

        for (let i = BARLEN * FIRST; i < len1; i += P_GAP) {
            s_off.push(i - 2)
            s_on.push(i)
        }

        return [ s_on, s_off ]
    }

    function sieve1(v, i) {
        const rotators = { 2: 5, 3: 1, 5: 3, 6: 0, 8: 4, 9: 2, 11: 6 }

        const GAP  = 128 // 16 bars
        const LPAD = 1024 // 128 bars

        const m = (48 + v) % 12
        const n = null
        const s = (i - LPAD) / GAP

        function rotato(x) {
            if (s < x) { return v }
            if (s < (13 - x)) { return n }
            if (s < (14 + x)) { return -v }
            if (s < (27 - x)) { return n }
            return v
        }

        return rotato(rotators[m])
    }

    const [ s_on, s_off ] = pedal1(n)

    return m1
        .keep(len1)
        .if(!n)
            .then(s => s.replaceIndices(-1, null))
        .mapPitch(sieve1)
        .transpose(84 - 12 * n)
        .toMelody()
        .withDuration(rhy1)
        .withVolume(vol1)
        .withStartTick(n * TICKS * 14)
        .withEventBefore(s_on, 'sustain', 1)
        .withEventAfter(s_off, 'sustain', 0)
        .tap(s => logTrack(n, 'part1', s))
}

function melody2(n) {
    log('entering m2', n)

    const toplen  = (n ? 37 : 43.5) * (8 - n)
    const len2    = [ 64, 413, 786, 1055 ][n] + toplen
    const delay2  = [ 500, 364, 190, 0 ][n]
    const dynamic = [ [ 85, 130 ], [ 60, V_FFF ], [ 35, V_FFF ], [ 10, V_FFF ] ][n]
    const adds   = n ? [ 0, 102, 230, 358, 1e9, 1e9 ] : [ 0, 48, 96, 136, 176, 208, 240 ]
    const trans  = [
        [ 10, 23, 16, 24, 23, 29, 25 ],
        [ 2, 19, 7, 17 ],
        [ 3, 17, 9, 7 ],
        [ 0, 0, 0, 12 ]
    ][n]

    function sieve2(seq, adds, trans, rev) {
        function mapper(v, i) {
            const m = (48 + v) % 12
            const n = null

            switch (m) {
            case 0:  return i >= adds[0] ? trans[0] + v : n
            case 11: return i >= adds[1] ? trans[1] - v : n
            case 2:  return i >= adds[2] ? trans[2] - v : n
            case 8:  return i >= adds[3] ? trans[3] - v : n
            case 5:  return i >= adds[4] ? trans[4] - v : n
            case 9:  return i >= adds[5] ? trans[5] - v : n
            case 3:  return i >= adds[6] ? trans[6] - v : n
            default: return n
            }
        }

        return rev ? seq.retrograde().mapPitch(mapper).retrograde() : seq.mapPitch(mapper)
    }

    function dynamics2() {
        return noteseq(imports.linear(len2, ...dynamic)).trim(0, V_FFF)
    }

    function v0_part2() {
        const volume = noteseq(imports.linear(toplen, V_FFF, 50))
        const adds = [ 0, 0, 0, 0, 0, 0, 0 ]
        const trans = [ 10, 23, 16, 24, 23, 29, 25 ]

        function repitch(p, i) {
            // Bb(low) -> B C -> C (one octave up); D -> C# (one octave up); E -> D (one octave up); F# -> D# (oou); G# -> E (oou)
            switch (p % 12) {
                case 10: return i >= 64 ? p + 1 : p
                case 0: return i >= 96 ? p + 12 : p
                case 2: return i >= 128 ? p + 11 : p
                case 4: return i >= 160 ? p + 10 : p
                case 6: return i >= 192 ? p + 9 : p
                case 8: return i >= 224 ? p + 8 : p
                default: {
                    dbg('error', p)
                    return null
                }
            }
        }

        function droplow(p, i) {
            const cmp = (i / toplen) * 81.5 - 20

            return p >= cmp ? p : null
        }

        return m1.drop(1945)
            .drop(toplen)
            .keep(toplen)
            .pipe(s => sieve2(s, adds, trans, 1))
            .mapPitch(repitch)
            .mapPitch(droplow)
            .toMelody()
            .withDuration(TICKS / 4)
            .withVolume(volume)
            .withEventBefore(0, 'lyric', 'Midpoint')
    }

    const r2 = rhythm(n)
    const v2 = dynamics2()

    return m1
        .drop(1945)
        .keep(len2)
        .pipe(s => sieve2(s, adds, trans))
        .if(!n).then(
            s => s.replaceIndices(24, 10)
                .replaceSlice(0, 12, seq => seq.withPitches([ 10, null, null, null, 22, null, null, null, 34, null, null, null ]))
                .replaceSlice(309, 325, s => s.repeat(3))
                .replaceSlice(286, 302, s => s.repeat(3))
                .dropRight(64)
        )
        .toMelody()
        .withDuration(r2)
        .withVolume(v2)
        .if(!n)
            .then(s => s.append(v0_part2()))
        .transpose(48)
        .pad({ pitch: [], duration: delay2 * TICKS, velocity: 0 })
        .tap(s => logTrack(n, 'part2', s))
}

function arpeggio(n) {
    function BARS(x) { return x * (8 - n) }

    let pitches
    let looplen
    let drops
    let desc

    switch (n) {
    case 0:
        return melody([])
    case 1:
        pitches = [ 53, 62, 69, 44 ]
        looplen = BARS(8)
        drops   = []
        desc    = 0
        break
    case 2:
        pitches = [ 66, 39, 47, 55 ]
        looplen = BARS(10)
        desc    = BARS(6)
        drops   = [ [ 66, BARS(6) ] ]
        break
    case 3:
        pitches = [ 37, 48, 58, 64 ]
        looplen = BARS(12)
        desc    = BARS(12)
        drops   = [ [ 64, BARS(8) ] ]
        break
    }

    return noteseq(pitches).loop(looplen)
        .replaceNth(10, null, 5)
        .replaceNth(5, null, 3)
        .while(() => drops.length)
            .do(s => {
                const [ pitch, loc ] = drops.pop()

                return s.replaceSlice(0, looplen - loc,
                    seq => seq.replaceIf(v => v.val() % 12 === pitch % 12, null))
            })
        .toMelody()
        .withDuration(rhythm(n))
        .withVolume(V_FFF)
        .withEventBefore(0, 'lyric', 'Arpeggios')
        .prepend(descension(n).keep(desc).retrograde())
        .withEventBefore(0, 'lyric', 'Section 2 -> 3')
        .tap(s => logTrack(n, 'climax', s))
}

function descension(n) {
    const max   = [ 0, 77, 79, 76 ][n]
    const min   = [ 0, 29, 31, 24 ][n]
    const drop  = [ [], [ 8, 7, 6, 3 ], [ 8, 5, 3, 8 ], [ 8, 6, 9, 1 ] ][n]
    const BARS  = 10
    const DEC   = BARS * 16
    const len   = (n ? 18 : 40) * BARS
    const rests = [ 1, 0, 3, 0, 1, 0, 4, 0, 2, 0, 1, 0, 2, 0, 3, 0, 1, 0 ]
    const ret   = []

    let seq
    let curr = [ 0, 65, 79, 48 ][n]

    if (n === 0) {
        return melody([])
    }

    let newmax
    for (let i = 0; i < len; i++) {
        newmax = max - Math.round(48 * i / DEC)

        let rest = (i + curr + newmax) % rests.length

        ret[i] = imports.constant(rests[rest] + 1, null)

        if (curr >= min && curr <= newmax && newmax - min >= (26 - n * 2)) {
            ret[i][0] = curr
        }

        curr -= drop[i % 4]

        if (curr < min) { curr += 48 }

        while (curr > newmax) {
            curr -= 12
        }
    }

    dbg(`Loop complete: min = ${min}, newmax = ${newmax}, oldmax = ${max}`)

    const volume = noteseq(imports.linear(len * 2, V_FFF, V_PP))

    return noteseq(ret.flat())
        .padRight(null, len * 2).keep(len * 2)
        .toMelody()
        .withDuration(rhythm(n))
        .withVolume(volume)
}

function melody3(n) {
    log('entering m3', n)

    return descension(n)
        .if(s => s.length)
            .then(s => s.withEventBefore(0, 'lyric', 'Section 3'))
        .tap(s => logTrack(n, 'part3', s))
}

function oct3ending(n) {
    function mechaprime(len, num) {
        const primes = imports.primes(num + 1).reverse()
        const gap    = primes.shift()
        const seqs   = primes.map((p, i) =>
            noteseq(imports.func(len, v => v % p ? 1 : 0))
                .pad(1, i * gap)
                .keep(len))

        return seqs[0].combineProduct(...seqs.slice(1))
    }

    const TEMPO0 = 40
    const TEMPO1 = 80
    const TEMPO2 = 100
    const TEMPO3 = 144

    const NUM_BEATS = 1192               // number of beats that might have a note
    const FIRST_BAR = 657 // 687         // bar at which this section begins
    const NUM_BARS  = 260 // 149         // number of bars in this section
    const START_BAR = NUM_BARS - 230     // bar at which we hit the main tempo
    const PART2_BAR = NUM_BARS - 150
    const CODA_BAR  = NUM_BARS - 93      // bar at which the coda begins
    const RIT_BAR   = NUM_BARS - 48      // bar at which the rit begins
    const TOT_BARS  = NUM_BARS - (n * 7) // number of bars in this part

    const barlen   = 8 - n
    const len      = TOT_BARS * barlen
    const loc_p2   = PART2_BAR * barlen
    const loc_coda = CODA_BAR * barlen
    const loc_fade = RIT_BAR * barlen

    const SILENCE   = [
        [ 0, 0 ],
        [ len - 999, len - 1141 ],
        [ len - 815, len - 939 ],
        [ len - 645, len - 753 ]
    ][n]

    const FIRST_TICK = FIRST_BAR * 2 * TICKS

    log('track', n, 'last(raw): len', len, 'coda len', len - loc_coda)

    const mek = mechaprime(len, 13).retrograde()

    const rhy = TICKS * 2 / barlen
    const vol = imports.constant(loc_coda - 18 * barlen, V_P)
        .concat(imports.linear(18 * barlen, V_P, V_F))
        .concat(imports.constant(64, V_FF))
        .concat(imports.linear(len - 64 - loc_coda, V_FF, V_PP))

    return m2
        .keepRight(NUM_BEATS)
        .padTo(null, len)
        .keepRight(len)
        .combine((a, b) => b.val() ? a : a.silence(), mek)
        .replaceIndices(len - 1178, null)
        .replaceIndices([ len - 814, len - 812, len - 473 ], v => v.transpose(12))
        .replaceSlice(...SILENCE, s => s.map(v => v.silence()))
        .transpose(47 + 12 * n)
        .toMelody()
        .withDuration(rhy)
        .withVolume(vol)
        .withStartTick(FIRST_TICK)
        .withEventBefore(-NUM_BEATS, 'sustain', 0)
        .withEventBefore(loc_coda, 'sustain', 1)
        .if(n===3)
            .then(s => s.replaceIndices(len - 735, e => e.setPitches([ 37, 103 ])))
            .then(s => s.replaceIndices(len - 724, e => e.setPitches([ 40, 104 ])))
        .if(DEBUG)
            .then(s => s.replaceSlice(0, loc_coda - 1,
                s => s.map((e, i) => e.isSilent() ? e : e.withEventBefore('text', String(len - i)))
            ))
        .addDelayAt(loc_p2, TICKS * 2)
        .withEventBefore(-1, 'text', 'release pedal after 8 - 12 seconds', { offset: TICKS * 2 * 6.125 + n * TICKS * 14 })
        .if(!n).then(m => {
            let loc = loc_coda - barlen

            for (let i = TEMPO2; i > TEMPO1; i -= 4) {
                m = m.withEventBefore(loc, 'tempo', i)

                loc -= 3 * barlen
            }

            loc = loc_fade + 2 * barlen

            for (let i = TEMPO3 - 4; i >= TEMPO2; i -= 4) {
                m = m.withEventBefore(loc, 'tempo', i)

                loc += 4 * barlen
            }

            return m
                .withEventBefore(START_BAR * barlen, 'tempo', TEMPO0)
                .withEventBefore(START_BAR * barlen, 'marker', 'Section 4')
                .withEventBefore(loc_p2, 'tempo', TEMPO1)
                .withEventBefore(loc_p2, 'marker', 'Section 5')
                .withEventBefore(loc_p2 - 15 * barlen, 'tempo', 36)
                .withEventBefore(loc_p2 - 17 * barlen, 'tempo', 38)
                .withEventBefore(loc_p2 - 19 * barlen, 'tempo', 40)
                .withEventBefore(loc_p2 - 21 * barlen, 'tempo', 42)
                .withEventBefore(loc_p2 - 23 * barlen, 'tempo', 44)
                .withEventBefore(loc_p2 - 25 * barlen, 'tempo', 46)
                .withEventBefore(loc_p2 - 29 * barlen, 'tempo', 48)
                .withEventBefore(loc_p2 - 37 * barlen, 'tempo', 46)
                .withEventBefore(loc_p2 - 45 * barlen, 'tempo', 44)
                .withEventBefore(loc_p2 - 56 * barlen, 'tempo', 42)
                .withEventBefore(loc_coda, 'tempo', TEMPO3)
                .withEventBefore(loc_coda, 'marker', 'Coda')
                .withEventBefore(loc_fade, 'text', 'poco a poco rit. al fine')
                .append(falling(FIRST_BAR), chords(FIRST_BAR))
        })
        .dropSlice(loc_p2 - 13 * barlen, loc_p2)
        .tap(s => logTrack(n, 'last', s))
}

function falling(bar) {
    const n     = SIB_IMPORT ? 1 : null
    const notes = [ 84, n, 72, 78, n, 66, 54, 66, 64, n, 52, 48, 46, 58, 72, 68, 44, 40, 36, n ]
    const tix   = TICKS * 2
    const rhy   = Math.round(rhythm(0) / 2 * (16 / notes.length))
    const mel   = noteseq(notes).toMelody().withDuration(rhy).withVolume(V_PP)

    bar *= tix

    return mel.transpose(15).withStartTick(bar + 38 * tix).append(
        mel.transpose(10).withStartTick(bar + 50 * tix),
        mel.transpose(5).withStartTick(bar + 53 * tix),
        mel.withStartTick(bar + 61 * tix),
        mel.transpose(-5).withStartTick(bar + 64 * tix),
        mel.transpose(-10).withStartTick(bar + 69 * tix),
        mel.transpose(-15).withStartTick(bar + 80 * tix),
        mel.keep(15).transpose(-20).withStartTick(bar + 87 * tix),
        mel.keep(9).transpose(-25).withStartTick(bar + 91 * tix),
        mel.keep(9).transpose(-30).withStartTick(bar + 93 * tix),
        mel.keep(6).transpose(-35).withStartTick(bar + 95 * tix)
    ).if(SIB_IMPORT)
        .then(s => s.map(e => e.max() < 1 ? e.setPitches([ 1 ]) : e))
}

function chords(bar) {
    const chord = [ 23, 24, 34, 35, 36 ]
    const rhy   = TICKS
    const tix   = TICKS * 2

    function getMel(loc) {
        chord[2]--
        return melody([ chord ])
            .withDuration(rhy)
            .withVolume(V_P + 5)
            .withStartTick((bar + loc) * tix)
    }

    return getMel(40)
        .append(getMel(45), getMel(53), getMel(57.5), getMel(66),
            getMel(70.5), getMel(78.5), getMel(83.5), getMel(96.5))
}

function glisses(bar) {
    const base = noteseq(imports.step(0, 127, 1))
        .toMelody()
        .withDuration(rhythm(0) / 6)
        .withVolume(V_PP)

    const tix = TICKS * 2

    bar *= tix

    function gliss(s1, s2, b) {
        if (s1 > s2) {
            return base.keepSlice(s2, s1).retrograde().withStartTick(bar + b * tix)
        }

        return base.keepSlice(s1, s2).withStartTick(bar + b * tix)
    }

    return gliss(42, 60, 26.75).append( // 18
        gliss(47, 62, 35.75), // 15
        gliss(63, 45, 39.75), // 18
        gliss(54, 66, 48.25), // 12
        gliss(69, 54, 52), // 15
        gliss(72, 54, 63.25), // 18
        gliss(61, 73, 68.25), // 12
        gliss(75, 60, 71.5), // 15
        gliss(77, 62, 77), // 15
        gliss(66, 78, 89.25), // 12
        gliss(71, 80, 96.25), // 9
    ).transpose(-12)
}

function cleanPianoIntervals([ a, b, c ], i) {
    function p_comp(a, b) {
        return Math.abs(a.pitch.max() - b.pitch.max()) > MAXGAP
    }

    function all_p_comp(a, b, c) {
        return p_comp(a, b) + p_comp(a, c) + p_comp(b, c)
    }

    if (a.isSilent() || b.isSilent() || c.isSilent()) { return [ a, b, c ] }

    if (all_p_comp(a, b, c) !== 3) { return [ a, b, c ] }

    let patch, change, saved_pitch

    if (b.pitch.max() > a.pitch.max() && b.pitch.max() > c.pitch.max()) {
        saved_pitch = c.pitch.max()

        if (DEBUG) {
            patch = p => c = c.setPitches(p).withEventBefore('text', `${saved_pitch}->${p}`)
        } else {
            patch = p => c = c.setPitches(p)
        }
    } else {
        saved_pitch = b.pitch.max()

        if (DEBUG) {
            patch = p => b = b.setPitches(p).withEventBefore('text', `${saved_pitch}->${p}`)
        } else {
            patch = p => b = b.setPitches(p)
        }
    }

    let done = 0
    let v = saved_pitch

    while (v >= 12) {
        v -= 12

        dbg(`note ${i}: trying ${saved_pitch} (${a.max()},${b.max()},${c.max()}) to ${v}`)

        patch(v)

        if (all_p_comp(a, b, c) !== 3) {
            done = 1
            break
        }
    }

    if (!done) {
        v = saved_pitch

        do {
            v += 12

            dbg(`note ${i}: trying ${saved_pitch} (${a.max()},${b.max()},${c.max()}) to ${v}`)

            patch(v)

        } while (all_p_comp(a, b, c) === 3)
    }

    dbg(`note ${i}: changed ${saved_pitch} to ${v}`)

    return [ a, b, c ]
}

function makeTrack(n) {
    log('Creating track', n)

    // Should really have been done based on ticks
    const ffn = [
        (e, i) => i > 2800 && i < 3200 && e.min() === 34,
        (e, i) => i > 2400 && i < 3100 && e.min() === 26,
        (e, i) =>
               (i > 2550 && i < 3100 && e.min() === 27)
            || (i > 2950 && i < 3100 && e.min() === 35),
        (e, i) =>
               (i > 2000 && i < 3300 && e.min() === 24 && i !== 2121 && i !== 3055)
            || (i > 2440 && i < 3300 && e.min() === 25)
            || (i > 2850 && i < 3100 && e.min() === 34)
            || (i > 2900 && i < 3200 && e.min() === 36)
    ][n]

    return melody1(n)
        .append(melody2(n))
        .append(arpeggio(n))
        .append(melody3(n))
        .append(oct3ending(n))
        .map((e, i) => ffn(e, i) ? e.silence() : e)
        .replaceIfWindow(3, 1, () => 1, cleanPianoIntervals)
        .if(SIB_IMPORT)
            .then(s => s.map((e, i) => e.isSilent() ? e.setPitches(1) : e))
        .withNewEvent('track-name', `Piano ${n + 1}`)
        .withNewEvent('instrument', 0)
        .withNewEvent('pan', 28 + 24 * n)
        .if(!n)
            .then(s => s.withNewEvents(tempoEvents()).withNewEvent('time-signature', '2/4'))
        .tap(s => logTrack(n, 'whole', s))
}

function makePlayOnStrings() {
    return glisses(657)
        .withInstrument('harp')
        .withNewEvent('pan', 52)
}

function makeScore() {
    const s = score([ 0, 1, 2, 3 ].map(makeTrack))
        .appendItems(makePlayOnStrings())
        .withTicksPerQuarter(TICKS)
        .withNewEvents(META)
        .writeMidi(__filename)
        .writeCanvas(__filename, { wd_scale: 2 })
        .expectHash(HASH)

    log('#Bytes:  ', s.toMidiBytes().length)
    log('Hash was:', s.toHash())
    log('Expected:', HASH)
}

makeScore()

const end = process.hrtime(start);

log('Time:    ', Math.round(1e3 * end[0] + Math.floor(end[1] / 1e6)), 'ms');
*/
