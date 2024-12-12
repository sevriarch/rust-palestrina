pub trait SequenceMember<PitchType: Clone + Copy> {
    fn new(value: PitchType) -> Box<Self>;
    fn pitches(&self) -> Vec<PitchType>;
    fn num_pitches(&self) -> usize;
    fn single_pitch(&self) -> Result<PitchType, &str>;
    fn is_silent(&self) -> bool;
    fn max(&self) -> Result<PitchType, &str>;
    fn min(&self) -> Result<PitchType, &str>;
    fn mean(&self) -> Result<PitchType, &str>;
    fn equals(&self, cmp: Self) -> bool;
    fn map_pitches(&self, f: fn(p: PitchType) -> PitchType) -> Result<Box<Self>, &str>;
    fn set_pitches(&self, p: Vec<PitchType>) -> Result<Box<Self>, &str>;
    fn silence(&self) -> Result<Box<Self>, &str>;
    fn invert(&self, p: PitchType) -> Result<Box<Self>, &str>;
    fn transpose(&self, p: PitchType) -> Result<Box<Self>, &str>;
    fn augment(&self, p: PitchType) -> Result<Box<Self>, &str>;
    fn diminish(&self, p: PitchType) -> Result<Box<Self>, &str>;
    fn modulus(&self, p: PitchType) -> Result<Box<Self>, &str>;

    // plus trim(),bounce(),scale(),gamut()
}
