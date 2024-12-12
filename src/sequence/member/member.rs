pub trait SequenceMember<MemberType: Clone + Copy, PitchType: Clone + Copy> {
    fn pitches(&self) -> Vec<PitchType>;
    fn num_pitches(&self) -> usize;
    fn single_pitch(&self) -> Option<PitchType>;
    fn is_silent(&self) -> bool;
    fn max(&self) -> Option<PitchType>;
    fn min(&self) -> Option<PitchType>;
    fn mean(&self) -> Option<PitchType>;
    fn equals(&self, cmp: Self) -> bool;
    fn map_pitches(&self, f: fn(p: PitchType) -> Option<PitchType>);
    fn set_pitches(&self) -> Option<Self>;
    fn silence(&self) -> Option<Self>;
    fn invert(&self) -> Self;
    fn transpose(&self) -> Self;
    fn augment(&self) -> Self;
    fn diminish(&self) -> Option<Self>;
    fn modulus(&self) -> Self;

    // plus trim(),bounce(),scale(),gamut()
}