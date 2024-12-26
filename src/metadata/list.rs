use crate::metadata::data::{Metadata, MetadataError};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataList {
    contents: Vec<Metadata>,
}

impl MetadataList {
    pub fn new(contents: Vec<Metadata>) -> Self {
        MetadataList { contents }
    }

    pub fn append(mut self, md: Metadata) -> Self {
        self.contents.push(md);
        self
    }

    /*
    pub fn augment_rhythm(self, by: i32) -> Result<Self, String> {
        for m in self.contents.iter() {
            m.timing.offset *= by;
        }

        Self
    }

    pub fn diminish_rhythm(self, by: i32) -> Result<Self, String> {
        if by == 0 {
            return Err("cannot diminish by 0".to_string());
        }

        Ok(self.mutate_each(|m| m.timing.offset /= by))
    }
    */
}

pub trait PushMetadata<T> {
    fn push(self, kind: &str, data: T) -> Result<Self, MetadataError>
    where
        Self: Sized;

    fn push_with_timing(
        self,
        kind: &str,
        data: T,
        tick: Option<u32>,
        offset: i32,
    ) -> Result<Self, MetadataError>
    where
        Self: Sized;
}

macro_rules! push_impl {
    ($($type:ty)*) => ($(
impl PushMetadata<$type> for MetadataList {
    fn push(mut self, kind: &str, data: $type) -> Result<Self, MetadataError>
    where
        Self: Sized,
    {
        let data = Metadata::try_from((kind, data))?;

        self.contents.push(data);

        Ok(self)
    }

    fn push_with_timing(
        mut self,
        kind: &str,
        data: $type,
        tick: Option<u32>,
        offset: i32
    ) -> Result<Self, MetadataError>
    where
        Self: Sized,
    {
        let mut data = Metadata::try_from((kind, data))?;

        if let Some(tick) = tick {
            data = data.with_exact_tick(tick);
        }

        if offset != 0 {
            data = data.with_offset(offset);
        }

        self.contents.push(data);

        Ok(self)
    }
}
    )*)
}

push_impl! { &str i16 f32 bool }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push() {
        assert!(MetadataList::default().push("foo", "bar").is_err());
    }
}
