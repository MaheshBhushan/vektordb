use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad file: {0}")]
    Corrupt(String),
    #[error("dimension mismatch: store has {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("vector id {0} out of range")]
    IdOutOfRange(u64),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, Error>;
