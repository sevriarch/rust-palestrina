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

    fn indices(&self, indices: Vec<i32>) -> Vec<Result<usize, &str>> {
        indices.into_iter().map(|i| self.index(i)).collect()
    }

    fn val_at(&self, index: i32) -> Result<T, &str> {
        let ix = self.index(index)?;

        Ok(self.clone_contents()[ix])
    }

    fn find_first_index(&self, f: fn(mem: T) -> bool) -> Result<usize, &str> {
        let contents = self.cts();

        for (i, item) in contents.iter().enumerate().take(self.length()) {
            if f(*item) {
                return Ok(i);
            }
        }

        Err("no matching index")
    }

    fn find_last_index(&self, f: fn(mem: T) -> bool) -> Result<usize, &str> {
        let contents = self.cts();
        let len = self.length();

        for (i, item) in contents.iter().rev().enumerate().take(len) {
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
        let last = self.index(end)?;

        println!("first {}, last {}", first, last);

        if last < first {
            return Err("last index was before first one");
        }

        let contents = self.cts();

        Ok(self.construct(contents[first..last].to_vec()))
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
        assert_eq!(coll.keep_slice(1, 3).unwrap().contents, vec![2, 3]);
        assert_eq!(coll.keep_slice(-4, -1).unwrap().contents, vec![3, 4, 5]);
        assert_eq!(coll.keep_slice(2, -4).unwrap().contents, vec![]);
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
