// MIDI event type bytes
pub const CONTROLLER_BYTE: u8 = 0xb0;

// MIDI controller IDs
pub const VOLUME_CONTROLLER: u8 = 0x07;
pub const BALANCE_CONTROLLER: u8 = 0x08;
pub const PAN_CONTROLLER: u8 = 0x0a;
pub const SUSTAIN_CONTROLLER: u8 = 0x40;

// Special on/off controller values
pub const EVENT_ON_VALUE: u8 = 0x7f;
pub const EVENT_OFF_VALUE: u8 = 0x00;

// Bytes associated with MIDI text events
pub const TEXT_EVENT: [u8; 2] = [0xff, 0x01];
pub const COPYRIGHT_EVENT: [u8; 2] = [0xff, 0x02];
pub const TRACK_NAME_EVENT: [u8; 2] = [0xff, 0x03];
pub const INSTRUMENT_NAME_EVENT: [u8; 2] = [0xff, 0x04];
pub const LYRIC_EVENT: [u8; 2] = [0xff, 0x05];
pub const MARKER_EVENT: [u8; 2] = [0xff, 0x06];
pub const CUE_POINT_EVENT: [u8; 2] = [0xff, 0x07];

// Bytes associated with other MIDI events
pub const TEMPO_EVENT: [u8; 3] = [0xff, 0x51, 0x03];
pub const TIME_SIGNATURE_EVENT: [u8; 3] = [0xff, 0x58, 0x04];
pub const KEY_SIGNATURE_EVENT: [u8; 3] = [0xff, 0x59, 0x02];
pub const END_TRACK_EVENT: [u8; 3] = [0xff, 0x2f, 0x00];

// Bytes associated with MIDI file structure
pub const HEADER_CHUNK: [u8; 4] = [0x4d, 0x54, 0x68, 0x64];
pub const HEADER_LENGTH: [u8; 4] = [0x00, 0x00, 0x00, 0x06];
pub const HEADER_FORMAT: [u8; 2] = [0x00, 0x01];
pub const TRACK_HEADER_CHUNK: [u8; 4] = [0x4d, 0x54, 0x72, 0x6b];
