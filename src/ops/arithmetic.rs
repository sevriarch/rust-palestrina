use num_traits::{Bounded, Num};

/// A trait that implements augment and diminish operations that permit
/// multiplying ints by floats and vice versa.
pub trait AugDim<MT>
where
    MT: Copy + Bounded + Num,
{
    /// Augment the target value by the value of self.
    fn augment_target(&self, v: &mut MT);

    /// Diminish the target value by the value of self.
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn augment_target() {
        let mut v_float = 2.5;

        2.augment_target(&mut v_float);
        assert_eq!(v_float, 5.0);

        2.5.augment_target(&mut v_float);
        assert_eq!(v_float, 12.5);

        let mut v_int = 2;

        2.augment_target(&mut v_int);
        assert_eq!(v_int, 4);

        2.5.augment_target(&mut v_int);
        assert_eq!(v_int, 10);
    }

    #[test]
    fn diminish_target() {
        let mut v_float = 12.5;

        2.diminish_target(&mut v_float);
        assert_eq!(v_float, 6.25);

        2.5.diminish_target(&mut v_float);
        assert_eq!(v_float, 2.5);

        let mut v_int = 10;

        2.diminish_target(&mut v_int);
        assert_eq!(v_int, 5);

        2.5.diminish_target(&mut v_int);
        assert_eq!(v_int, 2);
    }
}
