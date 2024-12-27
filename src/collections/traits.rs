// Traits common to all collections within Palestrina

use std::collections::hash_map::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[macro_export]
macro_rules! default_collection_methods {
    ($type:ty) => {
        fn cts(&self) -> Vec<$type> {
            self.contents.clone()
        }

        fn length(&self) -> usize {
            self.contents.len()
        }

        fn mutate_each<F: Fn(&mut $type)>(&mut self, f: F) -> &Self {
            for m in self.contents.iter_mut() {
                f(m);
            }
            self
        }

        fn mutate_contents<F: FnOnce(&mut Vec<$type>)>(mut self, f: F) -> Self {
            f(&mut self.contents);
            self
        }

        fn mutate_contents_result<F: FnOnce(&mut Vec<$type>) -> Result<(), String>>(
            mut self,
            f: F,
        ) -> Result<Self, String> {
            match f(&mut self.contents) {
                Err(str) => Err(str),
                Ok(()) => Ok(self),
            }
        }

        fn map(mut self, f: fn($type) -> $type) -> Self {
            self.contents = self.contents.into_iter().map(f).collect();
            self
        }

        fn replace_contents<F: FnOnce(Vec<$type>) -> Vec<$type>>(mut self, f: F) -> Self {
            self.contents = f(self.contents);
            self
        }

        fn with_contents(mut self, contents: Vec<$type>) -> Self {
            self.contents = contents;
            self
        }
    };
}

pub trait Collection<T: Clone + Debug>: Sized {
    fn new(contents: Vec<T>) -> Self;

    fn mutate_each<F: Fn(&mut T)>(&mut self, f: F) -> &Self;
    fn mutate_contents<F: FnOnce(&mut Vec<T>)>(self, f: F) -> Self;
    fn mutate_contents_result<F: FnOnce(&mut Vec<T>) -> Result<(), String>>(
        self,
        f: F,
    ) -> Result<Self, String>;
    fn replace_contents<F: FnOnce(Vec<T>) -> Vec<T>>(self, f: F) -> Self;
    fn with_contents(self, contents: Vec<T>) -> Self;
    fn map(self, f: fn(T) -> T) -> Self;

    fn cts(&self) -> Vec<T>;
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

        Ok(self.cts()[ix].clone())
    }

    fn find_first_index(&self, f: fn(mem: &T) -> bool) -> Option<usize> {
        for (i, item) in self.cts().iter().enumerate().take(self.length()) {
            if f(item) {
                return Some(i);
            }
        }

        None
    }

    fn find_last_index(&self, f: fn(mem: &T) -> bool) -> Option<usize> {
        let len = self.length();

        for (i, item) in self.cts().iter().rev().enumerate().take(len) {
            if f(item) {
                return Some(len - i - 1);
            }
        }

        None
    }

    fn find_indices(&self, f: fn(mem: &T) -> bool) -> Vec<usize> {
        let contents = self.cts();

        (0..self.length()).filter(|i| f(&contents[*i])).collect()
    }

    fn keep_slice(self, start: i32, end: i32) -> Result<Self, String> {
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

    fn keep(self, num: usize) -> Result<Self, String> {
        let len = self.length();
        let last = if num > len { len } else { num };

        Ok(self.mutate_contents(|c| {
            c.truncate(last);
        }))
    }

    fn keep_right(self, num: usize) -> Result<Self, String> {
        let len = self.length();
        let first = len.saturating_sub(num);

        Ok(self.mutate_contents(|c| {
            c.drain(..first);
        }))
    }

    fn keep_indices(self, indices: &[i32]) -> Result<Self, String> {
        let ix = self.indices(indices)?;

        Ok(self.mutate_contents(|c| {
            *c = ix.iter().map(|i| c[*i].clone()).collect();
        }))
    }

    fn keep_nth(self, n: usize) -> Result<Self, String> {
        if n == 0 {
            Err("cannot keep every 0th member".to_string())
        } else {
            Ok(self.replace_contents(|c| c.into_iter().step_by(n).collect()))
        }
    }

    fn drop_slice(self, start: i32, end: i32) -> Result<Self, String> {
        let first = self.index(start)?;
        let last = if end == self.length() as i32 {
            end as usize
        } else {
            self.index(end)?
        };

        if last < first {
            return Err("last index was before first one".to_string());
        }

        Ok(self.replace_contents(|c| {
            let mut ret = c[0..first].to_vec();
            ret.append(&mut c[last..].to_vec());

            ret
        }))
    }

    fn drop(self, num: usize) -> Result<Self, String> {
        let len = self.length();
        let first = if num > len { len } else { num };

        Ok(self.mutate_contents(|c| {
            c.drain(..first);
        }))
    }

    fn drop_right(self, num: usize) -> Result<Self, String> {
        let len = self.length();
        let last = len.saturating_sub(num);

        Ok(self.mutate_contents(|c| {
            c.truncate(last);
        }))
    }

    fn drop_indices(self, indices: &[i32]) -> Result<Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix.iter() {
                println!("removing index {}", *i);
                c.remove(*i);
            }
        }))
    }

    fn drop_nth(self, n: usize) -> Result<Self, String> {
        if n == 0 {
            Err("cannot keep every 0th member".to_string())
        } else {
            Ok(self.replace_contents(|c| {
                c.iter()
                    .enumerate()
                    .filter_map(|(i, v)| if i % n == 0 { None } else { Some(v.clone()) })
                    .collect()
            }))
        }
    }

    fn empty(self) -> Result<Self, String> {
        Ok(self.mutate_contents(|c| c.truncate(0)))
    }

    fn filter(&self, f: fn(&T) -> bool) -> Self {
        let newcontents = self.cts().into_iter().filter(f).collect();

        self.construct(newcontents)
    }

    fn insert_before(self, indices: &[i32], values: &[T]) -> Result<Self, String> {
        let ix = self.indices_sorted(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                c.splice(i..i, values.to_vec());
            }
        }))
    }

    fn insert_after(self, indices: &[i32], values: &[T]) -> Result<Self, String> {
        let ix = self.indices_sorted(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                c.splice(i + 1..i + 1, values.to_vec());
            }
        }))
    }

    fn replace_indices(self, indices: &[i32], values: &[T]) -> Result<Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                c.splice(i..i + 1, values.to_vec());
            }
        }))
    }

    fn mutate_indices(self, indices: &[i32], f: impl Fn(&mut T)) -> Result<Self, String> {
        let ix = self.indices(indices)?;

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                f(&mut c[i]);
            }
        }))
    }

    fn replace_first(self, finder: fn(&T) -> bool, val: T) -> Result<Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = val;
            }))
        } else {
            Ok(self)
        }
    }

    fn replace_last(self, finder: fn(&T) -> bool, val: T) -> Result<Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = val;
            }))
        } else {
            Ok(self)
        }
    }

    fn map_indices(self, indices: &[i32], f: fn(&T) -> T) -> Result<Self, String> {
        let mut ix = self.indices(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                c[i] = f(&c[i]);
            }
        }))
    }

    fn map_first(self, finder: fn(&T) -> bool, f: fn(&T) -> T) -> Result<Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = f(&c[i]);
            }))
        } else {
            Ok(self)
        }
    }

    fn map_last(self, finder: fn(&T) -> bool, f: fn(&T) -> T) -> Result<Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c[i] = f(&c[i]);
            }))
        } else {
            Ok(self)
        }
    }

    fn flat_map_indices(self, indices: &[i32], f: fn(&T) -> Vec<T>) -> Result<Self, String> {
        let mut ix = self.indices_sorted(indices)?;

        ix.dedup();

        Ok(self.mutate_contents(|c| {
            for i in ix.into_iter() {
                c.splice(i..i + 1, f(&c[i]));
            }
        }))
    }

    fn flat_map_first(self, finder: fn(&T) -> bool, f: fn(&T) -> Vec<T>) -> Result<Self, String> {
        if let Some(i) = self.find_first_index(finder) {
            Ok(self.mutate_contents(|c| {
                c.splice(i..i + 1, f(&c[i]));
            }))
        } else {
            Ok(self)
        }
    }

    fn flat_map_last(self, finder: fn(&T) -> bool, f: fn(&T) -> Vec<T>) -> Result<Self, String> {
        if let Some(i) = self.find_last_index(finder) {
            Ok(self.mutate_contents(|c| {
                c.splice(i..i + 1, f(&c[i]));
            }))
        } else {
            Ok(self)
        }
    }

    fn append(self, coll: &Self) -> Self {
        self.mutate_contents(|c| c.append(&mut coll.cts()))
    }

    fn append_items(self, items: &[T]) -> Self {
        self.mutate_contents(|c| c.append(&mut items.to_vec()))
    }

    fn prepend(self, coll: &Self) -> Self {
        self.replace_contents(|mut c| {
            let mut cts = coll.cts();

            cts.append(&mut c);
            cts
        })
    }

    fn prepend_items(self, items: &[T]) -> Self {
        self.replace_contents(|mut c| {
            let mut cts = items.to_vec();

            cts.append(&mut c);
            cts
        })
    }

    fn retrograde(self) -> Result<Self, String> {
        Ok(self.mutate_contents(|c| {
            c.reverse();
        }))
    }

    fn swap(self, (i1, i2): (i32, i32)) -> Result<Self, String> {
        let ix1 = self.index(i1)?;
        let ix2 = self.index(i2)?;

        Ok(self.mutate_contents(|c| {
            c.swap(ix1, ix2);
        }))
    }

    fn swap_many(self, tup: &[(i32, i32)]) -> Result<Self, String> {
        /* TODO: figure out if there's a way to make this work
        self.mutate_contents_result(|c| {
            for (i1, i2) in tup.iter() {
                let ixes = self.indices(&[*i1, *i2])?;

                c.swap(ixes[0], ixes[1]);
            }

            Ok(())
        })
        */
        let ixes: Vec<(usize, usize)> = tup
            .iter()
            .map(|(i1, i2)| {
                let ix1 = self.index(*i1);
                let ix2 = self.index(*i2);

                match (ix1, ix2) {
                    (Ok(i1), Ok(i2)) => Ok((i1, i2)),
                    _ => Err("invalid index".to_string()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(self.mutate_contents(|c| {
            for (i1, i2) in ixes.iter() {
                c.swap(*i1, *i2);
            }
        }))
    }

    // TODO: is this needed?
    fn split_contents_at(&self, indices: Vec<i32>) -> Result<Vec<Vec<T>>, &str> {
        let ix = self.indices_inclusive(indices)?;

        let cts = self.cts();
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

        let cts = self.cts();
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
        let (p1, p2): (Vec<T>, Vec<T>) = self.cts().into_iter().partition(f);

        Ok((self.construct(p1), self.construct(p2)))
    }

    fn group_by<KeyType: Hash + Eq + PartialEq + Debug>(
        &self,
        f: fn(&T) -> KeyType,
    ) -> Result<HashMap<KeyType, Self>, &str> {
        let mut rets = HashMap::<KeyType, Vec<T>>::new();

        for m in self.cts().iter() {
            rets.entry(f(m)).or_default().push(m.clone());
        }

        Ok(rets
            .into_iter()
            .map(|(k, v)| (k, self.construct(v.to_vec())))
            .collect())
    }

    fn pipe<TPipe>(&self, f: fn(&Self) -> TPipe) -> TPipe {
        f(self)
    }
    //tap(),each()
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
    fn empty() {
        assert_eq!(
            TestColl::new(vec![2, 3, 4]).empty().unwrap(),
            TestColl::new(vec![])
        );
    }

    #[test]
    fn filter() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.filter(|v| v % 2 == 0);

        assert_eq!(new.contents, vec![0, 2, 4, 6]);
    }

    #[test]
    fn insert_before() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.insert_before(&[1, 4, -5, -1], &[7, 8]);

        assert_contents_eq!(new, vec![0, 7, 8, 7, 8, 2, 3, 4, 7, 8, 5, 7, 8, 6]);
    }

    #[test]
    fn insert_after() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.insert_after(&[1, -5, 4, -1], &[7, 8]);

        assert_contents_eq!(new, vec![0, 2, 7, 8, 7, 8, 3, 4, 5, 7, 8, 6, 7, 8]);
    }

    #[test]
    fn replace_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.replace_indices(&[1, -5, 4, -1], &[7, 8]);

        assert_contents_eq!(new, vec![0, 7, 8, 3, 4, 7, 8, 7, 8]);
    }

    #[test]
    fn mutate_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.mutate_indices(&[1, -5, 4, -1], |v| *v += 3);

        assert_contents_eq!(new, vec![0, 8, 3, 4, 8, 9]);
    }

    #[test]
    fn replace_first() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().replace_first(|v| *v > 10, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().replace_first(|v| *v % 2 == 1, 10);
        assert_contents_eq!(new, vec![0, 2, 10, 4, 5, 6]);
    }

    #[test]
    fn replace_last() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().replace_last(|v| *v > 10, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().replace_last(|v| *v % 2 == 1, 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 10, 6]);
    }

    #[test]
    fn map_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.map_indices(&[1, -5, 4, -1], |v| *v + 5);

        assert_contents_eq!(new, vec![0, 7, 3, 4, 10, 11]);
    }

    #[test]
    fn map_first() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().map_first(|v| *v > 10, |v| *v + 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().map_first(|v| *v % 2 == 1, |v| *v + 10);
        assert_contents_eq!(new, vec![0, 2, 13, 4, 5, 6]);
    }

    #[test]
    fn map_last() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().map_last(|v| *v > 10, |v| *v + 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll.clone().map_last(|v| *v % 2 == 1, |v| *v + 10);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 15, 6]);
    }

    #[test]
    fn flat_map_indices() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let new = coll.flat_map_indices(&[1, -5, 4, -1], |v| vec![*v, *v * 2, *v * *v]);

        assert_contents_eq!(new, vec![0, 2, 4, 4, 3, 4, 5, 10, 25, 6, 12, 36]);
    }

    #[test]
    fn flat_map_first() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll
            .clone()
            .flat_map_first(|v| *v > 10, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll
            .clone()
            .flat_map_first(|v| *v % 2 == 1, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 13, 3, -7, 4, 5, 6]);
    }

    #[test]
    fn flat_map_last() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let new = coll
            .clone()
            .flat_map_last(|v| *v > 10, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 5, 6]);

        let new = coll
            .clone()
            .flat_map_last(|v| v % 2 == 1, |v| vec![*v + 10, *v, *v - 10]);
        assert_contents_eq!(new, vec![0, 2, 3, 4, 15, 5, -5, 6]);
    }

    #[test]
    fn append() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let app = TestColl::new(vec![9, 11, 13]);

        assert_eq!(
            coll.append(&app),
            TestColl::new(vec![0, 2, 3, 4, 5, 6, 9, 11, 13])
        );
    }

    #[test]
    fn append_items() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(
            coll.append_items(&[9, 11, 13]),
            TestColl::new(vec![0, 2, 3, 4, 5, 6, 9, 11, 13])
        );
    }

    #[test]
    fn prepend() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);
        let app = TestColl::new(vec![9, 11, 13]);

        assert_eq!(
            coll.prepend(&app),
            TestColl::new(vec![9, 11, 13, 0, 2, 3, 4, 5, 6])
        );
    }

    #[test]
    fn prepend_items() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_eq!(
            coll.prepend_items(&[9, 11, 13]),
            TestColl::new(vec![9, 11, 13, 0, 2, 3, 4, 5, 6])
        );
    }

    #[test]
    fn retrograde() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert_contents_eq!(coll.retrograde(), vec![6, 5, 4, 3, 2, 0]);
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
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().swap_many(&[(0, 6)]).is_err());
        assert!(coll.clone().swap_many(&[(1, 2), (-7, 0)]).is_err());
        assert_contents_eq!(
            coll.clone().swap_many(&[(1, -3), (2, -1), (0, 1)]),
            vec![4, 0, 6, 2, 5, 3]
        );
    }

    #[test]
    fn split_at() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        assert!(coll.clone().split_at(vec![7]).is_err());
        assert!(coll.clone().split_at(vec![-7]).is_err());

        let ret = coll.clone().split_at(vec![]).unwrap();
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0].contents, vec![0, 2, 3, 4, 5, 6]);

        let ret = coll.clone().split_at(vec![-2]).unwrap();

        assert_eq!(ret.len(), 2);
        assert_eq!(ret[0].contents, vec![0, 2, 3, 4]);
        assert_eq!(ret[1].contents, vec![5, 6]);

        let ret = coll.clone().split_at(vec![0, -6, -1, 6]).unwrap();

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

    #[test]
    fn group_by() {
        let coll = TestColl::new(vec![0, 2, 3, 4, 5, 6]);

        let map = coll.group_by(|i| i % 3).unwrap();
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
}
