use std::f64::consts::PI;

fn is_prime(n: i64) -> bool {
    let max = (n as f64).sqrt() as i64 + 1;

    for i in 2..max {
        if n % i == 0 {
            return false;
        }
    }

    n > 1
}

pub fn primes(n: usize) -> Vec<i64> {
    let mut ret = Vec::with_capacity(n);

    if n == 0 {
        return ret;
    }

    let mut curr = 2;
    let mut count = 0;

    loop {
        if is_prime(curr) {
            ret.push(curr);
            count += 1;
            if count >= n {
                break;
            }
        }

        curr += 1;
    }

    ret
}

pub fn primepi(n: usize) -> Vec<i64> {
    let mut ret = Vec::with_capacity(n);
    let mut count = 0;

    for i in 0..n {
        if is_prime(i as i64) {
            count += 1;
        }

        ret.push(count);
    }

    ret
}

pub fn squares(n: usize) -> Vec<i64> {
    (0..n as i64).map(|v| v * v).collect()
}

pub fn sqrt_floor(n: usize) -> Vec<i64> {
    (0..n).map(|v| (v as f64).sqrt() as i64).collect()
}

pub fn sqrt_round(n: usize) -> Vec<i64> {
    (0..n)
        .map(|v| (v as f64).powf(0.5).round() as i64)
        .collect()
}

pub fn sqrt_ceil(n: usize) -> Vec<i64> {
    (0..n).map(|v| (v as f64).powf(0.5).ceil() as i64).collect()
}

pub fn bigomega(n: usize) -> Vec<i64> {
    let primes = &primes(n);

    (0..n)
        .map(|v| {
            let mut num = v as i64;
            let mut ct = 0;

            for prime in primes {
                while num >= *prime && num % *prime == 0 {
                    num /= *prime;

                    ct += 1;
                }
            }

            ct
        })
        .collect()
}

pub fn triangular(n: usize) -> Vec<i64> {
    let mut tot = 0;
    let mut ret = Vec::with_capacity(n);

    for i in 0..n {
        tot += i as i64;
        ret.push(tot);
    }

    ret
}

pub fn fibonacci(n: usize) -> Vec<i64> {
    match n {
        0 => vec![],
        1 => vec![0],
        _ => {
            let mut ret = Vec::with_capacity(n);

            ret.push(0);
            ret.push(1);

            for i in 2..n {
                ret.push(ret[i - 1] + ret[i - 2]);
            }

            ret
        }
    }
}

pub fn binary_runs(n: usize) -> Vec<i64> {
    (0..n)
        .map(|v| {
            let mut ct = 0;
            let mut last = (1 & v) == 1;
            let mut bit = 1;

            while bit <= v {
                let curr = (bit & v) == 0;

                if last != curr {
                    last = curr;
                    ct += 1;
                }

                bit <<= 1;
            }

            ct
        })
        .collect()
}

pub fn infinity(n: usize) -> Vec<i64> {
    let mut ret = Vec::with_capacity(n);

    if n > 0 {
        ret.push(0);
    }

    for i in 1..n {
        ret.push(match i % 2 {
            0 => -ret[i / 2],
            _ => 1 + ret[(i - 1) / 2],
        });
    }

    ret
}

pub fn infinity_rhythmic(n: usize) -> Vec<i64> {
    if n == 0 {
        return vec![];
    }

    let fiblen = (n as f64).log2().ceil() as usize + 5;

    let fib = fibonacci(fiblen);
    let bin = binary_runs(n);

    (0..n).map(|v| fib[4 + bin[v] as usize]).collect()
}

fn infinity_var1_index(n: usize) -> i64 {
    if n == 0 {
        return 0;
    }

    match n % 3 {
        0 => -infinity_var1_index(n / 3),
        1 => infinity_var1_index(n / 3) - 2,
        _ => infinity_var1_index(n / 3) - 1,
    }
}

pub fn infinity_var1(n: usize) -> Vec<i64> {
    (0..n).map(infinity_var1_index).collect()
}

fn infinity_var2_index(n: usize) -> i64 {
    if n == 0 {
        return 0;
    }

    match n % 3 {
        0 => -infinity_var2_index(n / 3),
        1 => infinity_var2_index(n / 3) - 3,
        _ => -2 - infinity_var2_index(n / 3),
    }
}

pub fn infinity_var2(n: usize) -> Vec<i64> {
    (0..n).map(infinity_var2_index).collect()
}

pub fn rasmussen(n: usize) -> Vec<i64> {
    let ppi = primepi(n);
    let primes = &primes(n);

    (0..n as i64)
        .map(|v| {
            let mut num = v;
            let mut ct = 0;

            for prime in primes {
                while num >= *prime && num % *prime == 0 {
                    num /= *prime;

                    ct += ppi[*prime as usize];
                }
            }

            ct
        })
        .collect()
}

pub fn my1(n: usize) -> Vec<i64> {
    let tri = triangular(n + 1);

    (1..=n).map(|v| tri[v] % (v as f64).sqrt() as i64).collect()
}

pub fn bitrev(n: u64) -> Vec<i64> {
    (0..n)
        .map(|v| {
            if v == 0 {
                return 0;
            }

            let mut ret = 0;
            let mut bit1 = 1 << v.ilog2();
            let mut bit2 = 1;

            while bit1 != 0 {
                if v & bit1 != 0 {
                    ret |= bit2;
                }

                bit1 >>= 1;
                bit2 <<= 1;
            }

            ret
        })
        .collect()
}

pub fn constant<T: Clone>(n: usize, v: T) -> Vec<T> {
    vec![v; n]
}

pub fn sinusoidal<T: Into<f64>>(n: usize, width: T, first_angle: T, last_angle: T) -> Vec<i64> {
    let width: f64 = width.into();
    let last_angle: f64 = last_angle.into();
    let first_angle: f64 = first_angle.into();
    let gradiant = (last_angle - first_angle) / (n - 1) as f64;
    let pi180 = PI / 180.0;

    (0..n)
        .map(|v| {
            let angle: f64 = pi180 * (first_angle + (v as f64 * gradiant));

            (width * angle.sin()).round() as i64
        })
        .collect()
}

pub fn linear<T: Into<f64>>(n: usize, first_value: T, last_value: T) -> Vec<i64> {
    let first_value: f64 = first_value.into();
    let last_value: f64 = last_value.into();
    let gradiant = (last_value - first_value) / (n - 1) as f64;

    (0..n)
        .map(|v| (first_value + gradiant * v as f64).floor() as i64)
        .collect()
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::imports::*;

    #[test]
    fn test_primes() {
        assert_eq!(primes(0), vec![]);
        assert_eq!(primes(4), vec![2, 3, 5, 7]);
        assert_eq!(
            primes(32),
            vec![
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97, 101, 103, 107, 109, 113, 127, 131
            ]
        );
    }

    #[test]
    fn test_primepi() {
        assert_eq!(primepi(0), vec![]);
        assert_eq!(primepi(4), vec![0, 0, 1, 2]);
        assert_eq!(
            primepi(32),
            vec![
                0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9,
                9, 10, 10, 11
            ]
        );
    }

    #[test]
    fn test_squares() {
        assert_eq!(squares(0), vec![]);
        assert_eq!(squares(4), vec![0, 1, 4, 9]);
        assert_eq!(
            squares(32),
            vec![
                0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324,
                361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961
            ]
        );
    }

    #[test]
    fn test_sqrt_floor() {
        assert_eq!(sqrt_floor(0), vec![]);
        assert_eq!(sqrt_floor(4), vec![0, 1, 1, 1]);
        assert_eq!(
            sqrt_floor(32),
            vec![
                0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
                5, 5, 5, 5
            ]
        );
    }

    #[test]
    fn test_sqrt_round() {
        assert_eq!(sqrt_round(0), vec![]);
        assert_eq!(sqrt_round(4), vec![0, 1, 1, 2]);
        assert_eq!(
            sqrt_round(32),
            vec![
                0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 6
            ]
        );
    }

    #[test]
    fn test_sqrt_ceil() {
        assert_eq!(sqrt_ceil(0), vec![]);
        assert_eq!(sqrt_ceil(4), vec![0, 1, 2, 2]);
        assert_eq!(
            sqrt_ceil(32),
            vec![
                0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
                6, 6, 6, 6
            ]
        );
    }

    #[test]
    fn test_bigomega() {
        assert_eq!(bigomega(0), vec![]);
        assert_eq!(bigomega(4), vec![0, 0, 1, 1]);
        assert_eq!(
            bigomega(32),
            vec![
                0, 0, 1, 1, 2, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 2, 4, 1, 3, 1, 3, 2, 2, 1, 4, 2, 2, 3,
                3, 1, 3, 1
            ]
        );
    }

    #[test]
    fn test_triangular() {
        assert_eq!(triangular(0), vec![]);
        assert_eq!(triangular(4), vec![0, 1, 3, 6]);
        assert_eq!(
            triangular(32),
            vec![
                0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190,
                210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496
            ]
        );
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), vec![]);
        assert_eq!(fibonacci(1), vec![0]);
        assert_eq!(fibonacci(4), vec![0, 1, 1, 2]);
        assert_eq!(
            fibonacci(32),
            vec![
                0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
                6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040,
                1346269
            ]
        );
    }

    #[test]
    fn test_binary_runs() {
        assert_eq!(binary_runs(0), vec![]);
        assert_eq!(binary_runs(4), vec![0, 1, 2, 1]);
        assert_eq!(
            binary_runs(32),
            vec![
                0, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 2, 1, 2, 3, 4, 3, 4, 5, 4, 3, 2, 3, 4, 3,
                2, 3, 2, 1
            ]
        );
    }

    #[test]
    fn test_infinity() {
        assert_eq!(infinity(0), vec![]);
        assert_eq!(infinity(4), vec![0, 1, -1, 2]);
        assert_eq!(
            infinity(32),
            vec![
                0, 1, -1, 2, 1, 0, -2, 3, -1, 2, 0, 1, 2, -1, -3, 4, 1, 0, -2, 3, 0, 1, -1, 2, -2,
                3, 1, 0, 3, -2, -4, 5
            ]
        );
    }

    #[test]
    fn test_infinity_rhythmic() {
        assert_eq!(infinity_rhythmic(0), vec![]);
        assert_eq!(infinity_rhythmic(4), vec![3, 5, 8, 5]);
        assert_eq!(
            infinity_rhythmic(32),
            vec![
                3, 5, 8, 5, 8, 13, 8, 5, 8, 13, 21, 13, 8, 13, 8, 5, 8, 13, 21, 13, 21, 34, 21, 13,
                8, 13, 21, 13, 8, 13, 8, 5
            ]
        );
    }

    #[test]
    fn test_infinity_var1() {
        assert_eq!(infinity_var1(0), vec![]);
        assert_eq!(infinity_var1(4), vec![0, -2, -1, 2]);
        assert_eq!(
            infinity_var1(32),
            vec![
                0, -2, -1, 2, -4, -3, 1, -3, -2, -2, 0, 1, 4, -6, -5, 3, -5, -4, -1, -1, 0, 3, -5,
                -4, 2, -4, -3, 2, -4, -3, 0, -2
            ]
        );
    }

    #[test]
    fn test_infinity_var2() {
        assert_eq!(infinity_var2(0), vec![]);
        assert_eq!(infinity_var2(4), vec![0, -3, -2, 3]);
        assert_eq!(
            infinity_var2(32),
            vec![
                0, -3, -2, 3, -6, 1, 2, -5, 0, -3, 0, -5, 6, -9, 4, -1, -2, -3, -2, -1, -4, 5, -8,
                3, 0, -3, -2, 3, -6, 1, 0, -3
            ]
        );
    }

    #[test]
    fn test_rasmussen() {
        assert_eq!(rasmussen(0), vec![]);
        assert_eq!(rasmussen(4), vec![0, 0, 1, 2]);
        assert_eq!(
            rasmussen(32),
            vec![
                0, 0, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 6, 5, 5, 4, 7, 5, 8, 5, 6, 6, 9, 5, 6, 7, 6,
                6, 10, 6, 11
            ]
        );
    }

    #[test]
    fn test_my1() {
        assert_eq!(my1(0), vec![]);
        assert_eq!(my1(4), vec![0, 0, 0, 0]);
        assert_eq!(
            my1(32),
            vec![
                0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 3, 2, 2, 3, 1, 0, 0, 0, 1, 3, 1,
                0, 0, 1, 3
            ]
        );
    }

    #[test]
    fn test_bitrev() {
        assert_eq!(bitrev(0), vec![]);
        assert_eq!(bitrev(4), vec![0, 1, 1, 3]);
        assert_eq!(
            bitrev(32),
            vec![
                0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 13, 3, 11, 7, 15, 1, 17, 9, 25, 5, 21, 13, 29, 3,
                19, 11, 27, 7, 23, 15, 31
            ]
        );
    }

    #[test]
    fn test_constant() {
        assert_eq!(constant(0, 1.5), vec![]);
        assert_eq!(constant(1, 6), vec![6]);
        assert_eq!(constant(3, 1.5), vec![1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_sinusoidal() {
        assert_eq!(sinusoidal(2, 10, 0, 0), vec![0, 0]);
        assert_eq!(sinusoidal(2, 10, 0, 180), vec![0, 0]);
        assert_eq!(sinusoidal(3, 10, 0, 180), vec![0, 10, 0]);
        assert_eq!(sinusoidal(5, 10, 0, 360), vec![0, 10, 0, -10, 0]);
        assert_eq!(sinusoidal(5, 10, 90, 450), vec![10, 0, -10, 0, 10]);
        assert_eq!(
            sinusoidal(9, 40, 0, 360),
            vec![0, 28, 40, 28, 0, -28, -40, -28, 0]
        );
        assert_eq!(
            sinusoidal(9, 80, 45, 225),
            vec![57, 74, 80, 74, 57, 31, 0, -31, -57]
        );
        assert_eq!(
            sinusoidal(9, 80.0, 225.0, 45.0),
            vec![-57, -31, 0, 31, 57, 74, 80, 74, 57]
        );
    }

    #[test]
    fn test_linear() {
        assert_eq!(linear(2, 1, 1), vec![1, 1]);
        assert_eq!(linear(2, 1, 2), vec![1, 2]);
        assert_eq!(linear(3, 1, 0), vec![1, 0, 0]);
        assert_eq!(linear(3, 0, 1), vec![0, 0, 1]);
        assert_eq!(linear(3, -1, 0), vec![-1, -1, 0]);
        assert_eq!(
            linear(11, 10, 25),
            vec![10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25]
        );
    }
}
