// Traits common to all collections within Palestrina

use std::collections::hash_map::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[macro_export]
macro_rules! default_collection_methods {
    ($type:ty) => {
        fn clone_contents(&self) -> Vec<$type> {
            self.contents.clone()
        }

        fn length(&self) -> usize {
            self.contents.len()
        }

        // Call closure to mutate each member of self.contents, return ref to self
        fn mutate_each<F: Fn(&mut $type)>(&mut self, f: F) -> &Self {
            self.contents.iter_mut().for_each(f);
            self
        }

        // Call closure to mutate self.contents, return ref to self
        fn mutate_contents<F: FnOnce(&mut Vec<$type>)>(&mut self, f: F) -> &Self {
            f(&mut self.contents);
            self
        }

        // Call closure to mutate self.contents, return result containing ref to self
        fn mutate_contents_with_result<F: FnOnce(&mut Vec<$type>) -> Result<(), String>>(
            &mut self,
            f: F,
        ) -> Result<&Self, String> {
            match f(&mut self.contents) {
                Err(str) => Err(str),
                Ok(()) => Ok(self),
            }
        }

        fn map(mut self, f: fn($type) -> $type) -> Self {
            self.contents = self.contents.into_iter().map(f).collect();
            self
        }

        // Call closure to calculate new contents, replace existing contents, return ref to self
        fn replace_contents<F: FnOnce(&mut Vec<$type>) -> Vec<$type>>(&mut self, f: F) -> &Self {
            self.contents = f(&mut self.contents);
            self
        }

        // Set value of contents, return ref to self
        fn with_contents(&mut self, contents: Vec<$type>) -> &Self {
            self.contents = contents;
            self
        }

        // Return a reference to contents
        fn cts_ref(&self) -> &Vec<$type> {
            &self.contents
        }

        // Iterate (without consuming) over each member of this entity, return ref to self
        fn each<F: FnMut(&$type)>(&self, f: F) -> &Self {
            self.contents.iter().for_each(f);
            self
        }
    };
}

fn collection_index(i: i32, len: i32) -> Result<usize, String> {
    let ix = match i < 0 {
        false => i,
        true => len + i,
    };

    match ix < 0 || ix >= len {
        true => Err("index out of bounds".to_string()),
        false => Ok(ix as usize),
    }
}

pub trait Collection<T: Clone + Debug>: Sized {
    fn new(contents: Vec<T>) -> Self;

    fn mutate_each<F: Fn(&mut T)>(&mut self, f: F) -> &Self;
    fn mutate_contents<F: FnOnce(&mut Vec<T>)>(&mut self, f: F) -> &Self;
    fn mutate_contents_with_result<F: FnOnce(&mut Vec<T>) -> Result<(), String>>(
        &mut self,
        f: F,
    ) -> Result<&Self, String>;
    fn replace_contents<F: FnOnce(&mut Vec<T>) -> Vec<T>>(&mut self, f: F) -> &Self;
    fn with_contents(&mut self, contents: Vec<T>) -> &Self; // TODO: Probably should be in Sequence?
    fn cts_ref(&self) -> &Vec<T>;

    fn map(self, f: fn(T) -> T) -> Self;
    fn each<F: FnMut(&T)>(&self, f: F) -> &Self;

    fn clone_contents(&self) -> Vec<T>;
    fn length(&self) -> usize;
    fn construct(&self, contents: Vec<T>) -> Self;

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

    fn indices(&self, indices: &[i32]) -> Result<Vec<usize>, &str> {
        indices.iter().map(|i| self.index(*i)).collect()
    }

    // Return the supplied indices sorted from right to left
    fn indices_sorted(&self, indices: &[i32]) -> Result<Vec<usize>, &str> {
        let mut ix = self.indices(indices)?;

        ix.sort_unstable_by(|a, b| b.cmp(a));
        Ok(ix)
    }

    fn indices_inclusive(&self, indices: Vec<i32>) -> Result<Vec<usize>, &str> {
        indices
            .into_iter()
            .map(|i| self.index_inclusive(i))
            .collect()
    }

    fn val_at(&self, index: i32) -> Result<T, &str> {
        let ix = self.index(index)?;

        Ok(self.clone_contents()[ix].clone())
    }

    fn find_first_index(&self, f: fn(mem: &T) -> bool) -> Option<usize> {
        for (i, item) in self.clone_contents().iter().enumerate().take(self.length()) {
            if f(item) {
                return Some(i);
            }
        }

        None
    }

    fn find_last_index(&self, f: fn(mem: &T) -> bool) -> Option<usize> {
        let len = self.length();

        for (i, item) in self.clone_contents().iter().rev().enumerate().take(len) {
            if f(item) {
                return Some(len - i - 1);
            }
        }

        None
    }

    fn find_indices(&self, f: fn(mem: &T) -> bool) -> Vec<usize> {
        let contents = self.clone_contents();

        (0..self.length()).filter(|i| f(&contents[*i])).collect()
    }

    fn keep_slice(&mut self, start: i32, end: i32) -> Result<&Self, String> {
        let first = self.index(start)?;
        let last = self.index_inclusive(end)?;

        if last < first {
            return Err("last index was before first one".to_string());
        }

        Ok(self.mutate_contents(|c| {
            c.truncate(last);
            c.drain(..first);
        }))
    }

    fn keep(&mut self, num: usize) -> Result<&Self, String> {
        let len = self.length();
        let last = if num > len { len } else { num };

        Ok(self.mutate_contents(|c| {
            c.truncate(last);
        }))
    }

    fn keep_right(&mut self, num: usize) -> Result<&Self, String> {
        let len = self.length();
        let first = len.saturating_sub(num);

        Ok(self.mutate_contents(|c| {
            c.drain(..first);
        }))
    }

    fn keep_indices(&mut self, indices: &[i32]) -> Result<&Self, String> {
        let ix = self.indices(indices)?;

        Ok(self.replace_contents(|c| ix.iter().map(|i| c[*i].clone()).collect()))
    }

    fn keep_nth(&mut self, n: usize) -> Result<&Self, String> {
        if n == 0 {
            Err("cannot keep every 0th member".to_string())
        } else {
            // Replace rather than filter as unless n = 1 this is a much smaller vector
            Ok(self.replace_contents(|c| c.iter().step_by(n).cloned().collect()))
        }
    }

    fn drop_slice(&mut self, start: i32, end: i32) -> Result<&Self, String> {
        let first = self.index(start)?;
        let last = self.index_inclusive(end)?;

        if last < first {
            return Err("last index was before first one".to_string());
        }

        Ok(self.mutate_contents(|c| {
            c.drain(first..last);
        }))
    }

    fn drop(&mut self, num: usize) -> Result<&Self, String> {
        let len = self.length();
        let first = if num > len { len } else { num };

        Ok(self.mutate_contents(|c| {
            c.drain(..first);
        }))
    }

    fn drop_right(&mut self, num: usize) -> Result<&Self, String> {
        let len = self.length();
        let last = len.saturating_sub(num);

        Ok(self.mutate_contents(|c| {
            c.truncate(last);
        }))
    }

    fn drop_indices(&mut self, indices: &[i32]) -> Result<&Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c.remove(i);
            }
        }))
    }

    fn drop_nth(&mut self, n: usize) -> Result<&Self, String> {
        if n == 0 {
            return Err("cannot keep every 0th member".to_string());
        }

        Ok(self.mutate_contents(|c| {
            let mut ct = n - 1;
            c.retain(|_| {
                ct += 1;
                ct % n != 0
            });
        }))
    }

    fn mutate_slice(&mut self, start: i32, end: i32, f: impl Fn(&mut T)) -> Result<&Self, String> {
        let first = self.index(start)?;
        let last = self.index_inclusive(end)?;

        if last < first {
            return Err("last index was before first one".to_string());
        }

        Ok(self.mutate_contents(|c| {
            for m in c.iter_mut().take(last).skip(first) {
                f(m);
            }
        }))
    }

    fn empty(&mut self) -> Result<&Self, String> {
        Ok(self.mutate_contents(|c| c.truncate(0)))
    }

    // TODO: rewrite
    fn filter(&mut self, f: fn(&T) -> bool) -> &Self {
        self.mutate_contents(|c| {
            c.retain(f);
        })
    }

    fn insert_before(&mut self, indices: &[i32], values: &[T]) -> Result<&Self, String> {
        let ix = self.indices_sorted(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c.splice(i..i, values.to_vec());
            }
        }))
    }

    fn insert_after(&mut self, indices: &[i32], values: &[T]) -> Result<&Self, String> {
        let ix = self.indices_sorted(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c.splice(i + 1..i + 1, values.to_vec());
            }
        }))
    }

    fn replace_indices(&mut self, indices: &[i32], values: &[T]) -> Result<&Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c.splice(i..i + 1, values.to_vec());
            }
        }))
    }

    fn mutate_indices(&mut self, indices: &[i32], f: impl Fn(&mut T)) -> Result<&Self, String> {
        let ix = self.indices(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix {
                f(&mut c[i]);
            }
        }))
    }

    fn replace_first(&mut self, finder: fn(&T) -> bool, val: T) -> Result<&Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = val;
            }))
        } else {
            Ok(self)
        }
    }

    fn replace_last(&mut self, finder: fn(&T) -> bool, val: T) -> Result<&Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = val;
            }))
        } else {
            Ok(self)
        }
    }

    fn map_indices(&mut self, indices: &[i32], f: fn(&T) -> T) -> Result<&Self, String> {
        let mut ix = self.indices(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c[i] = f(&c[i]);
            }
        }))
    }

    fn map_first(&mut self, finder: fn(&T) -> bool, f: fn(&T) -> T) -> Result<&Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = f(&c[i]);
            }))
        } else {
            Ok(self)
        }
    }

    fn map_last(&mut self, finder: fn(&T) -> bool, f: fn(&T) -> T) -> Result<&Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = f(&c[i]);
            }))
        } else {
            Ok(self)
        }
    }

    fn flat_map_indices(&mut self, indices: &[i32], f: fn(&T) -> Vec<T>) -> Result<&Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix {
                c.splice(i..i + 1, f(&c[i]));
            }
        }))
    }

    fn flat_map_first(
        &mut self,
        finder: fn(&T) -> bool,
        f: fn(&T) -> Vec<T>,
    ) -> Result<&Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c.splice(i..i + 1, f(&c[i]));
            }))
        } else {
            Ok(self)
        }
    }

    fn flat_map_last(
        &mut self,
        finder: fn(&T) -> bool,
        f: fn(&T) -> Vec<T>,
    ) -> Result<&Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c.splice(i..i + 1, f(&c[i]));
            }))
        } else {
            Ok(self)
        }
    }

    fn append(&mut self, coll: &Self) -> &Self {
        self.mutate_contents(|c| c.append(&mut coll.clone_contents()))
    }

    fn append_items(&mut self, items: &[T]) -> &Self {
        self.mutate_contents(|c| c.append(&mut items.to_vec()))
    }

    fn prepend(&mut self, coll: &Self) -> &Self {
        self.replace_contents(|c| {
            let mut cts = coll.clone_contents();

            cts.append(c);
            cts
        })
    }

    fn prepend_items(&mut self, items: &[T]) -> &Self {
        self.replace_contents(|c| {
            let mut cts = items.to_vec();

            cts.append(c);
            cts
        })
    }

    fn retrograde(&mut self) -> Result<&Self, String> {
        Ok(self.mutate_contents(|c| {
            c.reverse();
        }))
    }

    fn swap(&mut self, (i1, i2): (i32, i32)) -> Result<&Self, String> {
        let ix1 = self.index(i1)?;
        let ix2 = self.index(i2)?;

        Ok(self.mutate_contents(|c| {
            c.swap(ix1, ix2);
        }))
    }

    fn swap_many(&mut self, tup: &[(i32, i32)]) -> Result<&Self, String> {
        let len = self.length() as i32;

        self.mutate_contents_with_result(|c| {
            for (i1, i2) in tup.iter() {
                let ix1 = collection_index(*i1, len)?;
                let ix2 = collection_index(*i2, len)?;

                c.swap(ix1, ix2);
            }

            Ok(())
        })
    }

    // TODO: is this needed?
    fn split_contents_at(&self, indices: Vec<i32>) -> Result<Vec<Vec<T>>, &str> {
        let ix = self.indices_inclusive(indices)?;

        let cts = self.clone_contents();
        let mut last: usize = 0;
        let mut ret: Vec<Vec<T>> = vec![];

        for i in ix {
            ret.push(cts[last..i].to_vec());
            last = i;
        }

        ret.push(cts[last..].to_vec());

        Ok(ret)
    }

    fn split_at(&self, indices: Vec<i32>) -> Result<Vec<Self>, &str> {
        let ix = self.indices_inclusive(indices)?;

        let cts = self.clone_contents();
        let mut last: usize = 0;
        let mut ret: Vec<Self> = vec![];

        for i in ix {
            ret.push(self.construct(cts[last..i].to_vec()));
            last = i;
        }

        ret.push(self.construct(cts[last..].to_vec()));

        Ok(ret)
    }

    fn partition(&self, f: fn(&T) -> bool) -> Result<(Self, Self), &str> {
        let (p1, p2): (Vec<T>, Vec<T>) = self.clone_contents().into_iter().partition(f);

        Ok((self.construct(p1), self.construct(p2)))
    }

    fn group_by<KeyType: Hash + Eq + PartialEq + Debug>(
        &self,
        f: fn(&T) -> KeyType,
    ) -> Result<HashMap<KeyType, Self>, &str> {
        let mut rets = HashMap::<KeyType, Vec<T>>::new();

        self.cts_ref().iter().for_each(|m| {
            rets.entry(f(m)).or_default().push(m.clone());
        });

        Ok(rets
            .into_iter()
            .map(|(k, v)| (k, self.construct(v.to_vec())))
            .collect())
    }

    fn pipe<TPipe>(&self, mut f: impl FnMut(&Self) -> TPipe) -> TPipe {
        f(self)
    }

    fn tap(&self, mut f: impl FnMut(&Self)) -> &Self {
        f(self);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::traits::Collection;

    macro_rules! assert_contents_eq {
        ($val: expr, $expected: expr) => {
            assert_eq!($val.unwrap().contents, $expected);
        };
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestColl {
        contents: Vec<i32>,
    }

    impl Collection<i32> for TestColl {
        default_collection_methods!(i32);

        fn new(contents: Vec<i32>) -> Self {
            TestColl { contents }
        }

        fn construct(&self, contents: Vec<i32>) -> Self {
            TestColl { contents }
        }
    }

    #[test]
    fn map() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.map(|v| v * v);

        assert_eq!(new.contents, vec![0, 4, 9, 16, 25, 36]);
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

        assert_eq!(coll.find_first_index(|v| *v % 2 == 1), Some(2));
        assert!(coll.find_first_index(|v| *v > 6).is_none())
    }

    #[test]
    fn find_last_index() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.find_last_index(|v| *v % 2 == 1), Some(4));
        assert!(coll.find_last_index(|v| *v > 6).is_none())
    }

    #[test]
    fn find_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.find_indices(|v| *v % 2 == 1), vec![2, 4]);
        assert_eq!(coll.find_indices(|v| *v > 6), vec![]);
    }

    #[test]
    fn keep_slice() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().keep_slice(0, -7).is_err());
        assert!(coll.clone().keep_slice(6, 0).is_err());
        assert!(coll.clone().keep_slice(4, -4).is_err());
        assert_contents_eq!(coll.clone().keep_slice(1, 6), vec![2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().keep_slice(-4, -1), vec![3, 4, 5]);
        assert_contents_eq!(coll.clone().keep_slice(2, -4), vec![]);
    }

    #[test]
    fn keep() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.clone().keep(0), vec![]);
        assert_contents_eq!(coll.clone().keep(5), vec![0, 2, 3, 4, 5]);
        assert_contents_eq!(coll.clone().keep(6), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().keep(7), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn keep_right() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.clone().keep_right(0), vec![]);
        assert_contents_eq!(coll.clone().keep_right(5), vec![2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().keep_right(6), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().keep_right(7), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn keep_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().keep_indices(&[0, 8]).is_err());
        assert_contents_eq!(coll.clone().keep_indices(&[]), vec![]);
        assert_contents_eq!(coll.clone().keep_indices(&[1, -1]), vec![2, 6]);
        assert_contents_eq!(
            coll.clone().keep_indices(&[0, 2, 3, -3, -1, 4, 5]),
            vec![0, 3, 4, 4, 6, 5, 6]
        );
    }

    #[test]
    fn keep_nth() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().keep_nth(0).is_err());
        assert_contents_eq!(coll.clone().keep_nth(1), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().keep_nth(3), vec![0, 4]);
    }

    #[test]
    fn drop_slice() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().drop_slice(0, -7).is_err());
        assert!(coll.clone().drop_slice(6, 0).is_err());
        assert!(coll.clone().drop_slice(4, -4).is_err());
        assert_contents_eq!(coll.clone().drop_slice(1, 6), vec![0]);
        assert_contents_eq!(coll.clone().drop_slice(-4, -1), vec![0, 2, 6]);
        assert_contents_eq!(coll.clone().drop_slice(2, -4), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn drop() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.clone().drop(0), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().clone().drop(5), vec![6]);
        assert_contents_eq!(coll.clone().drop(6), vec![]);
        assert_contents_eq!(coll.clone().drop(7), vec![]);
    }

    #[test]
    fn drop_right() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.clone().drop_right(0), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().drop_right(5), vec![0]);
        assert_contents_eq!(coll.clone().drop_right(6), vec![]);
        assert_contents_eq!(coll.clone().drop_right(7), vec![]);
    }

    #[test]
    fn drop_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().drop_indices(&[0, 8]).is_err());
        assert_contents_eq!(coll.clone().drop_indices(&[]), vec![0, 2, 3, 4, 5, 6]);
        assert_contents_eq!(coll.clone().drop_indices(&[1, 5, 1, -1]), vec![0, 3, 4, 5]);
    }

    #[test]
    fn drop_nth() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().drop_nth(0).is_err());
        assert_contents_eq!(coll.clone().drop_nth(1), vec![]);
        assert_contents_eq!(coll.clone().drop_nth(3), vec![2, 3, 5, 6]);
    }

    #[test]
    fn mutate_slice() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().mutate_slice(-1, 2, |v| *v += 10).is_err());
        assert_contents_eq!(
            coll.clone().mutate_slice(2, -1, |v| *v += 10),
            vec![0, 2, 13, 14, 15, 6]
        );
    }

    #[test]
    fn empty() {
        assert_eq!(
            TestColl::new(vec![2, 3, 4]).empty().unwrap(),
            &TestColl::new(vec![])
        );
    }

    #[test]
    fn filter() {
        assert_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).filter(|v| v % 2 == 0),
            &TestColl::new(vec![0, 2, 4, 6])
        );
    }

    #[test]
    fn insert_before() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).insert_before(&[1, 4, -5, -1], &[7, 8]),
            vec![0, 7, 8, 7, 8, 2, 3, 4, 7, 8, 5, 7, 8, 6]
        );
    }

    #[test]
    fn insert_after() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).insert_after(&[1, -5, 4, -1], &[7, 8]),
            vec![0, 2, 7, 8, 7, 8, 3, 4, 5, 7, 8, 6, 7, 8]
        );
    }

    #[test]
    fn replace_indices() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).replace_indices(&[1, -5, 4, -1], &[7, 8]),
            vec![0, 7, 8, 3, 4, 7, 8, 7, 8]
        );
    }

    #[test]
    fn mutate_indices() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).mutate_indices(&[1, -5, 4, -1], |v| *v += 3),
            vec![0, 8, 3, 4, 8, 9]
        );
    }

    #[test]
    fn replace_first() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.replace_first(|v| *v > 10, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.replace_first(|v| *v % 2 == 1, 10);
        assert_contents_eq!(new, vec![0, 2, 10, 4, 5, 6]);
    }

    #[test]
    fn replace_last() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.replace_last(|v| *v > 10, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.replace_last(|v| *v % 2 == 1, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 10, 6]);
    }

    #[test]
    fn map_indices() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).map_indices(&[1, -5, 4, -1], |v| *v + 5),
            vec![0, 7, 3, 4, 10, 11]
        );
    }

    #[test]
    fn map_first() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(
            coll.map_first(|v| *v > 10, |v| *v + 10),
            vec![0, 2, 3, 4, 5, 6]
        );

        assert_contents_eq!(
            coll.map_first(|v| *v % 2 == 1, |v| *v + 10),
            vec![0, 2, 13, 4, 5, 6]
        );
    }

    #[test]
    fn map_last() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(
            coll.map_last(|v| *v > 10, |v| *v + 10),
            vec![0, 2, 3, 4, 5, 6]
        );

        assert_contents_eq!(
            coll.map_last(|v| *v % 2 == 1, |v| *v + 10),
            vec![0, 2, 3, 4, 15, 6]
        );
    }

    #[test]
    fn flat_map_indices() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).flat_map_indices(&[1, -5, 4, -1], |v| vec![
                *v,
                *v * 2,
                *v * *v
            ]),
            vec![0, 2, 4, 4, 3, 4, 5, 10, 25, 6, 12, 36]
        );
    }

    #[test]
    fn flat_map_first() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.flat_map_first(|v| *v > 10, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.flat_map_first(|v| *v % 2 == 1, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 13, 3, -7, 4, 5, 6]);
    }

    #[test]
    fn flat_map_last() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.flat_map_last(|v| *v > 10, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.flat_map_last(|v| v % 2 == 1, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 15, 5, -5, 6]);
    }

    #[test]
    fn append() {
        assert_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).append(&TestColl::new(vec![9, 11, 13])),
            &TestColl::new(vec![0, 2, 3, 4, 5, 6, 9, 11, 13])
        );
    }

    #[test]
    fn append_items() {
        assert_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).append_items(&[9, 11, 13]),
            &TestColl::new(vec![0, 2, 3, 4, 5, 6, 9, 11, 13]),
        );
    }

    #[test]
    fn prepend() {
        assert_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).prepend(&TestColl::new(vec![9, 11, 13])),
            &TestColl::new(vec![9, 11, 13, 0, 2, 3, 4, 5, 6])
        );
    }

    #[test]
    fn prepend_items() {
        assert_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).prepend_items(&[9, 11, 13]),
            &TestColl::new(vec![9, 11, 13, 0, 2, 3, 4, 5, 6])
        );
    }

    #[test]
    fn retrograde() {
        assert_contents_eq!(
            TestColl::new(vec![0, 2, 3, 4, 5, 6]).retrograde(),
            vec![6, 5, 4, 3, 2, 0]
        );
    }

    #[test]
    fn swap() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().swap((0, 6)).is_err());
        assert!(coll.clone().swap((-7, 0)).is_err());
        assert_contents_eq!(coll.clone().swap((0, -1)), vec![6, 2, 3, 4, 5, 0]);
    }

    #[test]
    fn swap_many() {
        let mut coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().swap_many(&[(0, 6)]).is_err());
        assert!(coll.clone().swap_many(&[(1, 2), (-7, 0)]).is_err());

        assert_contents_eq!(coll.clone().swap_many(&[(1, -3)]), vec![0, 4, 3, 2, 5, 6]);
        assert_contents_eq!(
            coll.swap_many(&[(1, -3), (2, -1), (0, 1)]),
            vec![4, 0, 6, 2, 5, 3]
        );
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
        let (p1, p2) = TestColl::new(vec![0, 2, 3, 4, 5, 6])
            .partition(|i| i % 2 == 0)
            .unwrap();

        assert_eq!(p1.contents, vec![0, 2, 4, 6]);
        assert_eq!(p2.contents, vec![3, 5]);
    }

    #[test]
    fn group_by() {
        let map = TestColl::new(vec![0, 2, 3, 4, 5, 6])
            .group_by(|i| i % 3)
            .unwrap();

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&0).unwrap().contents, vec![0, 3, 6]);
        assert_eq!(map.get(&1).unwrap().contents, vec![4]);
        assert_eq!(map.get(&2).unwrap().contents, vec![2, 5]);
    }

    #[test]
    fn pipe() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(coll.pipe(|c| c.length()), 6);
    }

    #[test]
    fn tap() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let mut testvar = 1;

        coll.tap(|c| testvar = c.length());
    }

    #[test]
    fn each() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let mut tot = 0;

        coll.each(|m| tot += m);

        assert_eq!(tot, 20);
    }
}
