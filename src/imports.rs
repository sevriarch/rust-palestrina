fn is_prime(n: i64) -> bool {
    let max = (n as f64).sqrt() as i64 + 1;

    for i in 2..max {
        println!("testing {} against {} (max {})", n, i, max);
        if n % i == 0 { return false; }
    }

    n > 1
}

pub fn primes(n: usize) -> Vec<i64> {
    let mut ret = Vec::with_capacity(n);

    if n == 0 { return ret; }

    let mut curr = 2;
    let mut count = 0;

    loop {
        if is_prime(curr) {
            ret.push(curr);
            count += 1;
            if count >= n { break; }
        }

        curr += 1;
    }

    ret
}

pub fn infinity(n: usize) -> Vec<i64> {
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
    use crate::imports::*;

    #[test]
    fn test_primes() {
        assert_eq!(primes(0), vec![]);
        assert_eq!(primes(4), vec![2,3,5,7]);
        assert_eq!(primes(32), vec![2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131]);
    }

    #[test]
    fn test_infinity() {
        assert_eq!(infinity(0), vec![]);
        assert_eq!(infinity(4), vec![0,1,-1,2]);
        assert_eq!(infinity(32), vec![0,1,-1,2,1,0,-2,3,-1,2,0,1,2,-1,-3,4,1,0,-2,3,0,1,-1,2,-2,3,1,0,3,-2,-4,5]);
    }
}
