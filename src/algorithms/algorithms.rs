use num_traits::Num;
use std::cmp::PartialOrd;
use std::fmt::Debug;

pub fn invert<'a, T: Copy + Num>(pitch: &'a T) -> Box<dyn Fn(&mut T) + 'a> {
    Box::new(|v| *v = *pitch + *pitch - *v)
}

pub fn transpose<'a, T: Copy + Num>(pitch: &'a T) -> Box<dyn Fn(&mut T) + 'a> {
    Box::new(|v| *v = *pitch + *v)
}

pub fn augment<'a, T: Copy + Num>(mult: &'a T) -> Box<dyn Fn(&mut T) + 'a> {
    Box::new(|v| *v = *mult * *v)
}

pub fn diminish<'a, T: Copy + Num>(div: &'a T) -> Result<Box<dyn Fn(&mut T) + 'a>, String> {
    match div.is_zero() {
        true => Err("cannot divide by zero".to_string()),
        false => Ok(Box::new(|v| *v = *v / *div)),
    }
}

pub fn modulus<'a, T: Copy + Num + PartialOrd>(
    m: &'a T,
) -> Result<Box<dyn Fn(&mut T) + 'a>, String> {
    if m.is_zero() {
        return Err("modulus(): cannot modulus by zero".to_string());
    }

    Ok(Box::new(|v| {
        let ret = *v % *m;

        *v = match ret < T::zero() {
            true => ret + *m,
            false => ret,
        }
    }))
}

pub fn trim<'a, 'b, T: Num + Copy + PartialOrd + Debug>(
    a: Option<&'a T>,
    b: Option<&'a T>,
) -> Result<Box<dyn Fn(&mut T) + 'a>, String> {
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
                if *v > *max {
                    *v = *max;
                } else if *v < *min {
                    *v = *min;
                }
            }))
        }
        (Some(min), None) => Ok(Box::new(|v| {
            if *v < *min {
                *v = *min
            }
        })),
        (None, Some(max)) => Ok(Box::new(|v| {
            if *v > *max {
                *v = *max
            }
        })),
        (None, None) => Ok(Box::new(|_| {})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use assert_float_eq::assert_f32_near;

    #[test]
    fn test_invert() {
        let mut v = 12;
        invert(&5)(&mut v);
        assert_eq!(v, -2);

        let mut v = 4.2;
        invert(&1.6)(&mut v);
        assert_f32_near!(v, -1.0);
    }

    #[test]
    fn test_transpose() {
        let mut v = 12;
        transpose(&5)(&mut v);
        assert_eq!(v, 17);

        let mut v = 4.2;
        transpose(&1.6)(&mut v);
        assert_f32_near!(v, 5.8);
    }

    #[test]
    fn test_augment() {
        let mut v = 12;
        augment(&5)(&mut v);
        assert_eq!(v, 60);

        let mut v = 4.2;
        augment(&1.6)(&mut v);
        assert_f32_near!(v, 6.72);
    }

    #[test]
    fn test_diminish() {
        assert!(diminish(&0).is_err());
        assert!(diminish(&0.0).is_err());

        let mut v = 12;
        diminish(&5).unwrap()(&mut v);
        assert_eq!(v, 2);

        let mut v = 4.2;
        diminish(&1.6).unwrap()(&mut v);
        assert_f32_near!(v, 2.625);
    }

    #[test]
    fn test_modulus() {
        assert!(modulus(&0).is_err());
        assert!(modulus(&0.0).is_err());

        let mut v = 12;
        modulus(&5).unwrap()(&mut v);
        assert_eq!(v, 2);

        let mut v = 4.2;
        modulus(&1.6).unwrap()(&mut v);
        assert_f32_near!(v, 1.0);
    }

    #[test]
    fn test_trim() {
        assert!(trim(Some(&10), Some(&5)).is_err());

        let maxonlyi = trim(None, Some(&10)).unwrap();

        let mut v = 5;
        maxonlyi(&mut v);
        assert_eq!(v, 5);

        let mut v = 15;
        maxonlyi(&mut v);
        assert_eq!(v, 10);

        let maxonlyf = trim(None, Some(&10.5)).unwrap();

        let mut v = 5.5;
        maxonlyf(&mut v);
        assert_eq!(v, 5.5);

        let mut v = 15.5;
        maxonlyf(&mut v);
        assert_eq!(v, 10.5);

        let minonlyi = trim(Some(&10), None).unwrap();

        let mut v = 5;
        minonlyi(&mut v);
        assert_eq!(v, 10);

        let mut v = 15;
        minonlyi(&mut v);
        assert_eq!(v, 15);

        let minonlyf = trim(Some(&10.5), None).unwrap();

        let mut v = 5.5;
        minonlyf(&mut v);
        assert_eq!(v, 10.5);

        let mut v = 15.5;
        minonlyf(&mut v);
        assert_eq!(v, 15.5);

        let bothi = trim(Some(&2), Some(&10)).unwrap();

        let mut v = 0;
        bothi(&mut v);
        assert_eq!(v, 2);

        let mut v = 5;
        bothi(&mut v);
        assert_eq!(v, 5);

        let mut v = 15;
        bothi(&mut v);
        assert_eq!(v, 10);

        let bothf = trim(Some(&2.5), Some(&10.5)).unwrap();

        let mut v = 0.5;
        bothf(&mut v);
        assert_eq!(v, 2.5);

        let mut v = 5.5;
        bothf(&mut v);
        assert_eq!(v, 5.5);

        let mut v = 15.5;
        bothf(&mut v);
        assert_eq!(v, 10.5);
    }
}
