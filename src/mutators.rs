use num_traits::Num;
use std::cmp::PartialOrd;

pub fn invert<'a, T: Copy + Num>(pitch: &'a T) -> Result<Box<dyn Fn(T) -> T + 'a>, &str> {
    Ok(Box::new(move |v| *pitch + *pitch - v))
}

pub fn transpose<'a, T: Copy + Num>(pitch: &'a T) -> Result<Box<dyn Fn(T) -> T + 'a>, &str> {
    Ok(Box::new(move |v| *pitch + v))
}

pub fn augment<'a, T: Copy + Num>(mult: &'a T) -> Result<Box<dyn Fn(T) -> T + 'a>, &str> {
    Ok(Box::new(move |v| *mult * v))
}

pub fn diminish<'a, T: Copy + Num + From<i8>>(
    div: &'a T,
) -> Result<Box<dyn Fn(T) -> T + 'a>, &str> {
    match *div == T::from(0) {
        true => Err("cannot divide by zero"),
        false => Ok(Box::new(move |v| v / *div)),
    }
}

pub fn modulus<'a, T: Copy + Num + PartialOrd + From<i8>>(
    m: &'a T,
) -> Result<Box<dyn Fn(T) -> T + 'a>, &str> {
    if *m == T::from(0) {
        return Err("cannot modulus by zero");
    }

    Ok(Box::new(move |v| {
        let ret = v % *m;

        match ret < T::from(0) {
            true => ret + *m,
            false => ret,
        }
    }))
}

pub fn trim<'a, T: Num + Copy + PartialOrd>(
    a: &'a Option<T>,
    b: &'a Option<T>,
) -> Result<Box<dyn Fn(T) -> T + 'a>, &'a str> {
    match (a, b) {
        (Some(min), Some(max)) => {
            if min > max {
                return Err("trim(): minimum value cannot by higher than maximum value");
            }

            Ok(Box::new(|v| if v > *max { *max } else { v }))
        }
        (Some(min), None) => Ok(Box::new(|v| if v < *min { *min } else { v })),
        (None, Some(max)) => Ok(Box::new(|v| if v > *max { *max } else { v })),
        (None, None) => Ok(Box::new(|v| v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use assert_float_eq::assert_f32_near;

    #[test]
    fn test_invert() {
        assert_eq!(invert(&5).unwrap()(12), -7);
        assert_eq!(invert(&1.6).unwrap()(4.2), -1.0);
    }

    #[test]
    fn test_transpose() {
        assert_eq!(invert(&5).unwrap()(12), 17);
        assert_eq!(invert(&1.6).unwrap()(4.2), 5.8);
    }

    #[test]
    fn test_augment() {
        assert_eq!(augment(&5).unwrap()(12), 60);
        assert_f32_near!(augment(&1.6).unwrap()(4.2), 6.72);
    }

    #[test]
    fn test_diminish() {
        assert!(diminish(&0).is_err());
        assert!(diminish(&0.0).is_err());
        assert_eq!(diminish(&5).unwrap()(12), 2);
        assert_f32_near!(diminish(&1.6).unwrap()(4.2), 2.625);
    }

    #[test]
    fn test_modulus() {
        assert!(modulus(&0).is_err());
        assert!(modulus(&0.0).is_err());
        assert_eq!(modulus(&5).unwrap()(12), 2);
        assert_f32_near!(modulus(&1.6).unwrap()(4.2), 1.0);
    }
}
