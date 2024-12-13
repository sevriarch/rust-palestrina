// Traits common to all collections within Palestrina

pub trait Collection<T: Clone + Copy> {
    fn new(contents: Vec<T>) -> Self;

    fn cts(&self) -> Vec<T>;
    fn length(&self) -> usize;
    fn clone_contents(&self) -> Vec<T>;
    fn construct(&self, contents: Vec<T>) -> Box<Self>;

    fn index(&self, i: i32) -> Result<usize, &str> {
        let len = self.length() as i32;

        let ix = match i < 0 {
            false => i,
            true => len + i,
        };

        match ix < 0 || ix >= len {
            true => Err("index out of bounds"),
            false => Ok(ix as usize),
        }
    }

    fn index_inclusive(&self, i: i32) -> Result<usize, &str> {
        let len = self.length() as i32;

        let ix = match i < 0 {
            false => i,
            true => len + i,
        };

        match ix < 0 || ix > len {
            true => Err("index out of bounds"),
            false => Ok(ix as usize),
        }
    }

    fn indices(&self, indices: Vec<i32>) -> Result<Vec<usize>, &str> {
        indices.into_iter().map(|i| self.index(i)).collect()
    }

    fn indices_inclusive(&self, indices: Vec<i32>) -> Result<Vec<usize>, &str> {
        indices
            .into_iter()
            .map(|i| self.index_inclusive(i))
            .collect()
    }

    fn val_at(&self, index: i32) -> Result<T, &str> {
        let ix = self.index(index)?;

        Ok(self.clone_contents()[ix])
    }

    fn find_first_index(&self, f: fn(mem: T) -> bool) -> Result<usize, &str> {
        for (i, item) in self.cts().iter().enumerate().take(self.length()) {
            if f(*item) {
                return Ok(i);
            }
        }

        Err("no matching index")
    }

    fn find_last_index(&self, f: fn(mem: T) -> bool) -> Result<usize, &str> {
        let len = self.length();

        for (i, item) in self.cts().iter().rev().enumerate().take(len) {
            if f(*item) {
                return Ok(len - i - 1);
            }
        }

        Err("no matching index")
    }

    fn find_indices(&self, f: fn(mem: T) -> bool) -> Vec<usize> {
        let contents = self.cts();

        (0..self.length()).filter(|i| f(contents[*i])).collect()
    }

    fn keep_slice(&self, start: i32, end: i32) -> Result<Box<Self>, &str> {
        let first = self.index(start)?;
        let last = if end == self.length() as i32 {
            end as usize
        } else {
            self.index(end)?
        };

        if last < first {
            return Err("last index was before first one");
        }

        Ok(self.construct(self.cts()[first..last].to_vec()))
    }

    fn keep(&self, num: usize) -> Result<Box<Self>, &str> {
        let len = self.length();
        let last = if num > len { len } else { num };

        Ok(self.construct(self.cts()[..last].to_vec()))
    }

    fn keep_right(&self, num: usize) -> Result<Box<Self>, &str> {
        let len = self.length();
        let first = if num > len { 0 } else { len - num };

        Ok(self.construct(self.cts()[first..].to_vec()))
    }

    fn keep_indices(&self, indices: Vec<i32>) -> Result<Box<Self>, &str> {
        let ix = self.indices(indices)?;
        let contents = self.cts();

        Ok(self.construct(ix.into_iter().map(|i| contents[i]).collect()))
    }

    fn keep_nth(&self, n: usize) -> Result<Box<Self>, &str> {
        if n == 0 {
            Err("cannot keep every 0th member")
        } else {
            Ok(self.construct(self.cts().into_iter().step_by(n).collect()))
        }
    }

    fn drop_slice(&self, start: i32, end: i32) -> Result<Box<Self>, &str> {
        let first = self.index(start)?;
        let last = if end == self.length() as i32 {
            end as usize
        } else {
            self.index(end)?
        };

        if last < first {
            return Err("last index was before first one");
        }

        let cts = self.cts();
        let mut ret = cts[0..first].to_vec();
        ret.append(&mut cts[last..].to_vec());

        Ok(self.construct(ret))
    }

    fn drop(&self, num: usize) -> Result<Box<Self>, &str> {
        let len = self.length();
        let first = if num > len { len } else { num };

        Ok(self.construct(self.cts()[first..].to_vec()))
    }

    fn drop_right(&self, num: usize) -> Result<Box<Self>, &str> {
        let len = self.length();
        let last = if num > len { 0 } else { len - num };

        Ok(self.construct(self.cts()[..last].to_vec()))
    }

    fn drop_nth(&self, n: usize) -> Result<Box<Self>, &str> {
        if n == 0 {
            Err("cannot keep every 0th member")
        } else {
            Ok(self.construct(
                self.cts()
                    .into_iter()
                    .enumerate()
                    .filter(|&(i, _)| i % n != 0)
                    .map(|(_, v)| v)
                    .collect(),
            ))
        }
    }

    fn drop_indices(&self, indices: Vec<i32>) -> Result<Box<Self>, &str> {
        let ix = self.indices(indices)?;
        let contents = self.cts();

        Ok(self.construct(
            (0..self.length())
                .filter(|i| !ix.contains(i))
                .map(|i| contents[i])
                .collect(),
        ))
    }

    fn empty(&self) -> Box<Self> {
        self.construct(vec![])
    }

    fn filter(&self, f: fn(T) -> bool) -> Box<Self> {
        let newcontents = self
            .clone_contents()
            .into_iter()
            .filter(|m| f(*m))
            .collect();

        self.construct(newcontents)
    }

    fn map(&self, f: fn(T) -> T) -> Box<Self> {
        let newcontents = self.clone_contents().into_iter().map(f).collect();

        self.construct(newcontents)
    }

    fn split_at(&self, indices: Vec<i32>) -> Result<Vec<Box<Self>>, &str> {
        let ix = self.indices_inclusive(indices)?;

        let cts = self.cts();
        let mut last: usize = 0;
        let mut ret: Vec<Box<Self>> = vec![];

        for i in ix {
            ret.push(self.construct(cts[last..i].to_vec()));
            last = i;
        }

        ret.push(self.construct(cts[last..].to_vec()));

        Ok(ret)
    }

    fn partition(&self, f: fn(T) -> bool) -> Result<(Box<Self>, Box<Self>), &str> {
        let (p1, p2): (Vec<T>, Vec<T>) = self.cts().into_iter().partition(|v| f(*v));

        Ok((self.construct(p1), self.construct(p2)))
    }
}

/*
impl<T: Add<Output = T> + Copy> Add for CollectionMethods<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut newcontents = self.contents.clone();
        newcontents.append(&mut rhs.contents.clone());

        Self::new(newcontents)
    }
}
*/

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;

    macro_rules! assert_contents_eq {
        ($val: expr, $expected: expr) => {
            assert_eq!($val.unwrap().contents, $expected);
        };
    }

    #[derive(Debug, PartialEq)]
    struct TestColl {
        contents: Vec<i32>,
    }

    impl Collection<i32> for TestColl {
        fn new(contents: Vec<i32>) -> Self {
            Self { contents }
        }

        fn length(&self) -> usize {
            self.contents.len()
        }

        fn cts(&self) -> Vec<i32> {
            self.contents.clone()
        }

        fn clone_contents(&self) -> Vec<i32> {
            self.contents.clone()
        }

        fn construct(&self, contents: Vec<i32>) -> Box<Self> {
            Box::new(Self { contents })
        }
    }

    #[test]
    fn index() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.index(0), Ok(0));
        assert_eq!(coll.index(5), Ok(5));
        assert!(coll.index(6).is_err());
        assert_eq!(coll.index(-1), Ok(5));
        assert_eq!(coll.index(-6), Ok(0));
        assert!(coll.index(-7).is_err());
    }

    #[test]
    fn find_first_index() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.find_first_index(|v| v % 2 == 1), Ok(2));
        assert!(coll.find_first_index(|v| v > 6).is_err())
    }

    #[test]
    fn find_last_index() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.find_last_index(|v| v % 2 == 1), Ok(4));
        assert!(coll.find_last_index(|v| v > 6).is_err())
    }

    #[test]
    fn find_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.find_indices(|v| v % 2 == 1), vec![2, 4]);
        assert_eq!(coll.find_indices(|v| v > 6), vec![]);
    }

    #[test]
    fn keep_slice() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.keep_slice(0, -7).is_err());
        assert!(coll.keep_slice(6, 0).is_err());
        assert!(coll.keep_slice(4, -4).is_err());
        assert_contents_eq!(coll.keep_slice(1, 6), vec![2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.keep_slice(-4, -1), vec![3, 4, 5]);
        assert_contents_eq!(coll.keep_slice(2, -4), vec![]);
    }

    #[test]
    fn keep() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.keep(0), vec![]);
        assert_contents_eq!(coll.keep(5), vec![0, 2, 3, 4, 5]);
        assert_contents_eq!(coll.keep(6), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.keep(7), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn keep_right() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.keep_right(0), vec![]);
        assert_contents_eq!(coll.keep_right(5), vec![2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.keep_right(6), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.keep_right(7), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn keep_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.keep_indices(vec![0, 8]).is_err());
        assert_contents_eq!(coll.keep_indices(vec![]), vec![]);
        assert_contents_eq!(coll.keep_indices(vec![1, -1]), vec![2, 6]);
        assert_contents_eq!(
            coll.keep_indices(vec![0, 2, 3, -3, -1, 4, 5]),
            vec![0, 3, 4, 4, 6, 5, 6]
        );
    }

    #[test]
    fn keep_nth() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.keep_nth(0).is_err());
        assert_contents_eq!(coll.keep_nth(1), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.keep_nth(3), vec![0, 4]);
    }

    #[test]
    fn drop_slice() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.drop_slice(0, -7).is_err());
        assert!(coll.drop_slice(6, 0).is_err());
        assert!(coll.drop_slice(4, -4).is_err());
        assert_contents_eq!(coll.drop_slice(1, 6), vec![0]);
        assert_contents_eq!(coll.drop_slice(-4, -1), vec![0, 2, 6]);
        assert_contents_eq!(coll.drop_slice(2, -4), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn drop() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.drop(0), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.drop(5), vec![6]);
        assert_contents_eq!(coll.drop(6), vec![]);
        assert_contents_eq!(coll.drop(7), vec![]);
    }

    #[test]
    fn drop_right() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.drop_right(0), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.drop_right(5), vec![0]);
        assert_contents_eq!(coll.drop_right(6), vec![]);
        assert_contents_eq!(coll.drop_right(7), vec![]);
    }

    #[test]
    fn drop_nth() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.drop_nth(0).is_err());
        assert_contents_eq!(coll.drop_nth(1), vec![]);
        assert_contents_eq!(coll.drop_nth(3), vec![2, 3, 5, 6]);
    }

    #[test]
    fn filter() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.filter(|v| v % 2 == 0);

        assert_eq!(new.contents, vec![0, 2, 4, 6]);
    }

    #[test]
    fn map() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.map(|v| v * v);

        assert_eq!(new.contents, vec![0, 4, 9, 16, 25, 36]);
    }

    #[test]
    fn drop_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.drop_indices(vec![0, 8]).is_err());
        assert_contents_eq!(coll.drop_indices(vec![]), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.drop_indices(vec![1, -1]), vec![0, 3, 4, 5]);
    }

    #[test]
    fn split_at() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.split_at(vec![7]).is_err());
        assert!(coll.split_at(vec![-7]).is_err());

        let ret = coll.split_at(vec![]).unwrap();
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0].contents, vec![0, 2, 3, 4, 5, 6]);

        let ret = coll.split_at(vec![-2]).unwrap();

        assert_eq!(ret.len(), 2);
        assert_eq!(ret[0].contents, vec![0, 2, 3, 4]);
        assert_eq!(ret[1].contents, vec![5, 6]);

        let ret = coll.split_at(vec![0, -6, -1, 6]).unwrap();

        assert_eq!(ret.len(), 5);
        assert_eq!(ret[0].contents, vec![]);
        assert_eq!(ret[1].contents, vec![]);
        assert_eq!(ret[2].contents, vec![0, 2, 3, 4, 5]);
        assert_eq!(ret[3].contents, vec![6]);
        assert_eq!(ret[4].contents, vec![]);
    }

    #[test]
    fn partition() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let (p1, p2) = coll.partition(|i| i % 2 == 0).unwrap();
        assert_eq!(p1.contents, vec![0, 2, 4, 6]);
        assert_eq!(p2.contents, vec![3, 5]);
    }

    /*
    #[test]
    fn add() {
        let c1 = TestColl::new(vec![1, 5, 6]);
        let c2 = TestColl::new(vec![2, 7, 8, 9]);
        let c3 = TestColl::new(vec![1, 5, 6, 2, 7, 8, 9]);

        assert_eq!((c1 + c2).contents, c3.contents);
    }
    */
}
