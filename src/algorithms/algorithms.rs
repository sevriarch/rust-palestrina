use num_traits::Num;
use std::cmp::PartialOrd;
use std::fmt::Debug;

pub fn invert<'a, T: Copy + Num>(pitch: &'a T) -> Box<dyn Fn(&T) -> T + 'a> {
    Box::new(move |v| *pitch + *pitch - *v)
}

pub fn transpose<'a, T: Copy + Num>(pitch: &'a T) -> Box<dyn Fn(&T) -> T + 'a> {
    Box::new(move |v| *pitch + *v)
}

pub fn augment<'a, T: Copy + Num>(mult: &'a T) -> Box<dyn Fn(&T) -> T + 'a> {
    Box::new(move |v| *mult * *v)
}

pub fn diminish<'a, T: Copy + Num>(div: &'a T) -> Result<Box<dyn Fn(&T) -> T + 'a>, String> {
    match div.is_zero() {
        true => Err("cannot divide by zero".to_string()),
        false => Ok(Box::new(move |v| *v / *div)),
    }
}

pub fn modulus<'a, T: Copy + Num + PartialOrd>(
    m: &'a T,
) -> Result<Box<dyn Fn(&T) -> T + 'a>, String> {
    if m.is_zero() {
        return Err("modulus(): cannot modulus by zero".to_string());
    }

    Ok(Box::new(move |v| {
        let ret = *v % *m;

        match ret < T::zero() {
            true => ret + *m,
            false => ret,
        }
    }))
}

pub fn trim<'a, 'b, T: Num + Copy + PartialOrd + Debug>(
    a: Option<&'a T>,
    b: Option<&'a T>,
) -> Result<Box<dyn Fn(T) -> T + 'a>, String> {
    match (a, b) {
        (Some(min), Some(max)) => {
            if min > max {
                return Err(format!(
                    "trim(): minimum value {:?} cannot be higher than maximum value {:?}",
                    min, max
                )
                .to_string());
            }

            Ok(Box::new(|v| {
                if v > *max {
                    *max
                } else if v < *min {
                    *min
                } else {
                    v
                }
            }))
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
        assert_eq!(invert(&5)(&12), -2);
        assert_f32_near!(invert(&1.6)(&4.2), -1.0);
    }

    #[test]
    fn test_transpose() {
        assert_eq!(transpose(&5)(&12), 17);
        assert_f32_near!(transpose(&1.6)(&4.2), 5.8);
    }

    #[test]
    fn test_augment() {
        assert_eq!(augment(&5)(&12), 60);
        assert_f32_near!(augment(&1.6)(&4.2), 6.72);
    }

    #[test]
    fn test_diminish() {
        assert!(diminish(&0).is_err());
        assert!(diminish(&0.0).is_err());
        assert_eq!(diminish(&5).unwrap()(&12), 2);
        assert_f32_near!(diminish(&1.6).unwrap()(&4.2), 2.625);
    }

    #[test]
    fn test_modulus() {
        assert!(modulus(&0).is_err());
        assert!(modulus(&0.0).is_err());
        assert_eq!(modulus(&5).unwrap()(&12), 2);
        assert_f32_near!(modulus(&1.6).unwrap()(&4.2), 1.0);
    }

    #[test]
    fn test_trim() {
        assert!(trim(Some(&10), Some(&5)).is_err());

        let maxonlyi = trim(None, Some(&10)).unwrap();
        assert_eq!(maxonlyi(5), 5);
        assert_eq!(maxonlyi(15), 10);

        let maxonlyf = trim(None, Some(&10.5)).unwrap();
        assert_eq!(maxonlyf(5.5), 5.5);
        assert_eq!(maxonlyf(15.5), 10.5);

        let minonlyi = trim(Some(&10), None).unwrap();
        assert_eq!(minonlyi(5), 10);
        assert_eq!(minonlyi(15), 15);

        let minonlyf = trim(Some(&10.5), None).unwrap();
        assert_eq!(minonlyf(5.5), 10.5);
        assert_eq!(minonlyf(15.5), 15.5);

        let bothi = trim(Some(&2), Some(&10)).unwrap();
        assert_eq!(bothi(0), 2);
        assert_eq!(bothi(5), 5);
        assert_eq!(bothi(15), 10);

        let bothf = trim(Some(&2.5), Some(&10.5)).unwrap();
        assert_eq!(bothf(0.5), 2.5);
        assert_eq!(bothf(5.5), 5.5);
        assert_eq!(bothf(15.5), 10.5);
    }
}
