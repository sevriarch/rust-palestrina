use num_traits::Num;
use std::cmp::PartialOrd;
use std::fmt::Debug;

pub trait AugDim<MT> {
    fn augment_target(&self, v: &mut MT);
    fn diminish_target(&self, v: &mut MT);
}

macro_rules! single_conv_aug_dim {
    ($type:ident for $($target:ty)*) => ($(
        impl AugDim<$target> for $type {
            fn augment_target(&self, v: &mut $target) {
                *v *= (*self as $target);
            }

            fn diminish_target(&self, v: &mut $target) {
                *v /= (*self as $target);
            }
        }
    )*)
}

macro_rules! double_conv_aug_dim {
    ($type:ident for $($target:ty)*) => ($(
        impl AugDim<$target> for $type {
            fn augment_target(&self, v: &mut $target) {
                *v = ((*v as $type) * self) as $target;
            }

            fn diminish_target(&self, v: &mut $target) {
                *v = ((*v as $type) / self) as $target;
            }
        }
    )*)
}

macro_rules! make_double_conv_aug_dim {
    (for $($ty:ident)*) => ($(
        double_conv_aug_dim!($ty for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
    )*)
}

macro_rules! make_single_conv_aug_dim_int {
    (for $($ty:ident)*) => ($(
        single_conv_aug_dim!($ty for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
    )*)
}

macro_rules! make_single_conv_aug_dim_float {
    (for $($ty:ident)*) => ($(
        single_conv_aug_dim!($ty for f32 f64);
    )*)
}

make_single_conv_aug_dim_int!(for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128);
make_single_conv_aug_dim_float!(for usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64);
make_double_conv_aug_dim!(for f32 f64);

type Algorithm<'a, T> = Box<dyn Fn(&mut T) + 'a>;

pub fn invert<T: Copy + Num>(pitch: &T) -> Algorithm<T> {
    Box::new(|v| *v = *pitch + *pitch - *v)
}

pub fn transpose<T: Copy + Num>(pitch: &T) -> Algorithm<T> {
    Box::new(|v| *v = *pitch + *v)
}

pub fn augment<T, MT>(t: &MT) -> Algorithm<T>
where
    T: Copy + Num,
    MT: AugDim<T>,
{
    Box::new(|v| t.augment_target(v))
}

pub fn diminish<T, MT>(t: &MT) -> Result<Algorithm<T>, String>
where
    T: Copy + Num,
    MT: AugDim<T> + Num,
{
    match t.is_zero() {
        true => Err("cannot divide by zero".to_string()),
        false => Ok(Box::new(|v| t.diminish_target(v))),
    }
}

pub fn modulus<T: Copy + Num + PartialOrd>(m: &T) -> Result<Algorithm<T>, String> {
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

pub fn trim<'a, T: Num + Copy + PartialOrd + Debug>(
    a: Option<&'a T>,
    b: Option<&'a T>,
) -> Result<Algorithm<'a, T>, String> {
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
        let mut v = 13;
        augment(&7)(&mut v);
        assert_eq!(v, 91);
        augment(&1.6)(&mut v);
        assert_eq!(v, 145);

        let mut v = 4.2;
        augment(&1.6)(&mut v);
        assert_f32_near!(v, 6.72);
        augment(&2)(&mut v);
        assert_f32_near!(v, 13.44);
    }

    #[test]
    fn test_diminish() {
        assert!(diminish::<i32, i32>(&0).is_err());
        assert!(diminish::<f32, f32>(&0.0).is_err());

        let mut v = 12;
        diminish(&5).unwrap()(&mut v);
        assert_eq!(v, 2);
        diminish(&0.5).unwrap()(&mut v);
        assert_eq!(v, 4);

        let mut v = 4.2;
        diminish(&1.6).unwrap()(&mut v);
        assert_f32_near!(v, 2.625);
        diminish(&5).unwrap()(&mut v);
        assert_f32_near!(v, 0.525);
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
