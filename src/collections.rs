// Traits common to all collections within Palestrina

//use std::ops::Add;

pub trait CollectionMethods<T: Clone + Copy> {
    fn new(contents: Vec<T>) -> Self;
    fn clone_contents(&self) -> Vec<T>;
    fn construct(&self, contents: Vec<T>) -> Box<Self>;

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

    //fn retrograde(&self) -> Box<Self> { }
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
    use crate::collections::CollectionMethods;

    struct TestColl {
        contents: Vec<i32>,
    }

    impl CollectionMethods<i32> for TestColl {
        fn new(contents: Vec<i32>) -> Self {
            Self { contents }
        }

        fn clone_contents(&self) -> Vec<i32> {
            self.contents.clone()
        }

        fn construct(&self, contents: Vec<i32>) -> Box<Self> {
            Box::new(Self { contents })
        }
    }

    #[test]
    fn filter() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5]);
        let new = coll.filter(|v| v % 2 == 0);

        assert_eq!(new.contents, vec![0, 2, 4]);
    }

    #[test]
    fn map() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5]);
        let new = coll.map(|v| v * v);

        assert_eq!(new.contents, vec![0, 4, 9, 16, 25]);
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
