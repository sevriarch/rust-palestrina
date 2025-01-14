use anyhow::{anyhow, Result};
use num_traits::{Euclid, Num};
use std::cmp::PartialOrd;
use std::ops::SubAssign;

fn name_to_notes(name: &str) -> Result<Vec<i8>> {
    match name {
        "chromatic" => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "octatonic12" => Ok(vec![0, 1, 3, 4, 6, 7, 9, 10]),
        "octatonic21" => Ok(vec![0, 2, 3, 5, 6, 8, 9, 11]),
        "wholetone" => Ok(vec![0, 2, 4, 6, 8, 10]),
        "major" => Ok(vec![0, 2, 4, 5, 7, 9, 11]),
        "minor" => Ok(vec![0, 2, 3, 5, 7, 8, 10]),
        "ionian" => Ok(vec![0, 2, 4, 5, 7, 9, 11]),
        "dorian" => Ok(vec![0, 2, 3, 5, 7, 9, 10]),
        "phrygian" => Ok(vec![0, 1, 3, 5, 7, 8, 10]),
        "lydian" => Ok(vec![0, 2, 4, 6, 7, 9, 11]),
        "mixolydian" => Ok(vec![0, 2, 4, 5, 7, 9, 10]),
        "aeolian" => Ok(vec![0, 2, 3, 5, 7, 8, 10]),
        "locrian" => Ok(vec![0, 1, 3, 5, 6, 8, 10]),
        "pentatonic" => Ok(vec![0, 2, 4, 7, 9]),
        "pentatonicc" => Ok(vec![0, 2, 4, 7, 9]),
        "pentatonicd" => Ok(vec![0, 2, 5, 7, 10]),
        "pentatonice" => Ok(vec![0, 3, 5, 8, 10]),
        "pentatonicg" => Ok(vec![0, 2, 5, 7, 9]),
        "pentatonica" => Ok(vec![0, 3, 5, 7, 10]),
        _ => Err(anyhow!("unknown scale: {}", name)),
    }
}

#[derive(Clone, Debug)]
pub struct Scale<T>
where
    T: Copy + Num + From<i8> + TryInto<usize> + TryFrom<usize> + Euclid + PartialOrd + SubAssign,
{
    notes: Vec<T>,
    length: T,
    octave: T,
}

impl<
        T: Copy + Num + From<i8> + TryInto<usize> + TryFrom<usize> + Euclid + PartialOrd + SubAssign,
    > Default for Scale<T>
{
    fn default() -> Self {
        Self {
            notes: vec![],
            length: T::zero(),
            octave: T::from(12),
        }
    }
}

impl<
        T: Copy + Num + From<i8> + TryInto<usize> + TryFrom<usize> + Euclid + PartialOrd + SubAssign,
    > Scale<T>
{
    pub fn new() -> Self {
        Self {
            notes: vec![],
            length: T::zero(),
            octave: T::from(12),
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        let notes = name_to_notes(name)?;

        Ok(Self {
            notes: notes.into_iter().map(T::from).collect(),
            ..Default::default()
        })
    }

    pub fn with_octave(mut self, o: T) -> Self {
        self.octave = o;
        self
    }

    pub fn with_notes(mut self, notes: Vec<T>) -> Result<Self> {
        self.notes = notes;
        self.length = self
            .notes
            .len()
            .try_into()
            .map_err(|_| anyhow!("scale length too long"))?;

        Ok(self)
    }

    pub fn with_name(self, name: &str) -> Result<Self> {
        let notes = name_to_notes(name)?;

        self.with_notes(notes.into_iter().map(T::from).collect())
    }

    pub fn fit_to_scale<'a>(&'a self, zeroval: &'a T) -> Box<dyn Fn(&mut T) + 'a> {
        Box::new(|v| {
            let ix = v.rem_euclid(&self.length);
            let mut octaves = *v / self.length;

            if !ix.is_zero() && *v < T::zero() {
                octaves -= T::one();
            }

            // This should never happen, so defaulting to 0 should be safe
            let ix = ix.try_into().unwrap_or(0);

            *v = *zeroval + self.notes[ix] + octaves * self.octave;
        })
    }
    // impl Fn(T) -> T + use<'_, 'a, T> {
    /*
    pub fn fit_to_scale<'a>(&self, zeroval: &'a T) -> Box<dyn Fn(&T) -> T + 'a> {
        |v: T| {
            let ix = v.rem_euclid(&self.length);
            let mut octaves = v / self.length;
            if !ix.is_zero() && v < T::zero() {
                octaves -= T::one();
            }

            // This should never happen, so defaulting to 0 should be safe
            let ix = ix.try_into().unwrap_or(0);

            *zeroval + self.notes[ix] + octaves * self.octave
        }
    }
    */
}

#[cfg(test)]
mod tests {
    use crate::entities::scale::Scale;

    #[test]
    fn with_octave() {
        assert_eq!(Scale::<i32>::new().octave, 12);
        assert_eq!(Scale::new().with_octave(8).octave, 8);
    }

    #[test]
    fn from_name() {
        assert_eq!(Scale::<i32>::new().notes, vec![]);
        assert_eq!(
            Scale::<i32>::from_name("lydian").unwrap().notes,
            vec![0, 2, 4, 6, 7, 9, 11]
        );
        assert_eq!(
            Scale::<i32>::from_name("pentatonic").unwrap().notes,
            vec![0, 2, 4, 7, 9]
        );

        assert!(Scale::<i32>::from_name("turbofish").is_err());
    }

    #[test]
    fn with_name() {
        assert_eq!(Scale::<i32>::new().notes, vec![]);
        assert_eq!(
            Scale::<i32>::new().with_name("lydian").unwrap().notes,
            vec![0, 2, 4, 6, 7, 9, 11]
        );
        assert_eq!(
            Scale::<i32>::new().with_name("pentatonic").unwrap().notes,
            vec![0, 2, 4, 7, 9]
        );

        assert!(Scale::<i32>::new().with_name("turbofish").is_err());
    }

    #[test]
    fn with_notes() {
        assert_eq!(
            Scale::new()
                .with_notes(vec![0, 2, 3, 4, 5, 6, 9])
                .unwrap()
                .notes,
            vec![0, 2, 3, 4, 5, 6, 9]
        );
    }

    #[test]
    fn fit_to_scale() {
        let scale = Scale::<i32>::new().with_name("lydian").unwrap();

        let mut vec: Vec<i32> = (-20_i32..20_i32).collect::<Vec<i32>>();
        let f = scale.fit_to_scale(&60);

        for v in vec.iter_mut() {
            f(v);
        }

        assert_eq!(
            vec,
            vec![
                26, 28, 30, 31, 33, 35, 36, 38, 40, 42, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60,
                62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91, 93
            ]
        );

        let mut vec: Vec<i32> = vec![0, 2, 4];
        let f = scale.fit_to_scale(&0);

        for v in vec.iter_mut() {
            f(v);
        }

        assert_eq!(vec, vec![0, 4, 7]);
    }
}
