use crate::entities::timing::Timing;
use crate::metadata::data::{Metadata, MetadataError};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataList {
    pub contents: Vec<Metadata>,
}

impl MetadataList {
    pub fn new(contents: Vec<Metadata>) -> Self {
        MetadataList { contents }
    }

    // TODO: is this needed?
    pub fn append(&mut self, md: Metadata) -> &Self {
        self.contents.push(md);
        self
    }

    pub fn last_tick(&self, curr: u32) -> Result<u32, String> {
        if self.contents.is_empty() {
            return Ok(curr);
        }

        let mut max = curr;
        for m in self.contents.iter() {
            let c = m.timing.start_tick(curr)?;

            if c > max {
                max = c;
            }
        }

        Ok(max)
    }

    pub fn mutate_exact_tick(&mut self, f: impl Fn(&mut u32)) -> &Self {
        for m in self.contents.iter_mut() {
            m.timing.mutate_exact_tick(&f);
        }

        self
    }

    pub fn mutate_offset(&mut self, f: impl Fn(&mut i32)) -> &Self {
        for m in self.contents.iter_mut() {
            m.timing.mutate_offset(&f);
        }

        self
    }
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
            data.with_exact_tick(tick);
        }

        if offset != 0 {
            data.with_offset(offset);
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
    use crate::entities::timing::EventTiming;
    use crate::metadata::data::{Metadata, MetadataData};

    #[test]
    fn push() {
        assert!(MetadataList::default().push("foo", "bar").is_err());

        assert_eq!(
            MetadataList::default().push("text", "test text").unwrap(),
            MetadataList {
                contents: vec![Metadata::try_from(("text", "test text")).unwrap()]
            }
        );
    }

    #[test]
    fn push_with_timing() {
        assert!(MetadataList::default()
            .push_with_timing("foo", "bar", Some(50), 50)
            .is_err());

        assert_eq!(
            MetadataList::default()
                .push_with_timing("text", "test text", Some(50), 50)
                .unwrap(),
            MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(50),
                        offset: 50
                    }
                }]
            }
        );
    }

    #[test]
    fn last_tick() {
        assert_eq!(MetadataList::default().last_tick(0), Ok(0));
        assert_eq!(MetadataList::default().last_tick(128), Ok(128));

        assert_eq!(
            MetadataList {
                contents: vec![
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: None,
                            offset: 50
                        },
                    },
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: Some(100),
                            offset: 25
                        },
                    },
                ],
            }
            .last_tick(0),
            Ok(125)
        );

        assert_eq!(
            MetadataList {
                contents: vec![
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: None,
                            offset: 50
                        },
                    },
                    Metadata {
                        data: MetadataData::Text("test text".to_string()),
                        timing: EventTiming {
                            tick: Some(100),
                            offset: 50
                        },
                    },
                ],
            }
            .last_tick(200),
            Ok(250)
        );
    }

    #[test]
    fn mutate_exact_tick() {
        assert_eq!(
            MetadataList::default()
                .push_with_timing("text", "test text", None, 50)
                .unwrap()
                .mutate_exact_tick(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: None,
                        offset: 50
                    }
                }]
            }
        );

        assert_eq!(
            MetadataList::default()
                .push_with_timing("text", "test text", Some(20), 50)
                .unwrap()
                .mutate_exact_tick(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(40),
                        offset: 50
                    }
                }]
            }
        );
    }

    #[test]
    fn mutate_offset() {
        assert_eq!(
            MetadataList::default()
                .push_with_timing("text", "test text", Some(20), 50)
                .unwrap()
                .mutate_offset(|v| *v *= 2),
            &MetadataList {
                contents: vec![Metadata {
                    data: MetadataData::Text("test text".to_string()),
                    timing: EventTiming {
                        tick: Some(20),
                        offset: 100
                    }
                }]
            }
        );
    }
}
