pub fn infinity(n: usize) -> Vec<i32> {
    let mut ret = Vec::with_capacity(n);

    if n > 0 {
        ret.push(0);
    }

    for i in 1..n {
        ret.push(match i % 2 {
            0 => -ret[i / 2],
            1 => 1 + ret[(i - 1) / 2],
            _ => 0,
        });
    }

    ret
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::imports::infinity;

    #[test]
    fn test_infinity() {
        assert_eq!(infinity(0), vec![]);
        assert_eq!(infinity(4), vec![0,1,-1,2]);
        assert_eq!(infinity(32), vec![0,1,-1,2,1,0,-2,3,-1,2,0,1,2,-1,-3,4,1,0,-2,3,0,1,-1,2,-2,3,1,0,3,-2,-4,5]);
    }
}
