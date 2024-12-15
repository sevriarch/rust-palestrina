use num_traits::Num;

pub fn invert<T: Copy + Clone + Num>(pitch: &'static T) -> Box<dyn Fn(T) -> T> {
    Box::new(|v| *pitch + *pitch - v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert() {
        assert_eq!(invert(&5)(2), 8);
        assert_eq!(invert(&1.6)(4.2), (-1.0));
    }
}
