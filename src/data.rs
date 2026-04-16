use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use rwkv_tokenizer::WorldTokenizer;

const BINIDX_MAGIC: &[u8; 9] = b"MMIDIDX\0\0";

#[derive(Clone, Copy, Debug)]
pub enum DatasetFormat {
    Text,
    BinIdx,
}

#[derive(Debug)]
pub enum TrainingCorpus {
    Text(TextCorpus),
    BinIdx(BinIdxCorpus),
}

impl TrainingCorpus {
    pub fn load(path: &str, format: DatasetFormat, tokenizer: &WorldTokenizer) -> Result<Self> {
        match format {
            DatasetFormat::Text => Ok(Self::Text(TextCorpus::load(path, tokenizer)?)),
            DatasetFormat::BinIdx => Ok(Self::BinIdx(BinIdxCorpus::load(path)?)),
        }
    }

    pub fn total_tokens(&self) -> usize {
        match self {
            Self::Text(corpus) => corpus.total_tokens(),
            Self::BinIdx(corpus) => corpus.total_tokens(),
        }
    }

    pub fn sample_batch(
        &self,
        batch_size: usize,
        ctx_len: usize,
        step: usize,
        magic_prime: Option<u64>,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let total_tokens = self.total_tokens();
        if total_tokens <= ctx_len + 1 {
            bail!(
                "dataset is too small: need at least {} tokens, found {}",
                ctx_len + 2,
                total_tokens
            );
        }

        let mut inputs = Vec::with_capacity(batch_size * ctx_len);
        let mut targets = Vec::with_capacity(batch_size * ctx_len);

        for row in 0..batch_size {
            let offset = sample_offset(total_tokens, ctx_len, step, row, batch_size, magic_prime);
            let window = match self {
                Self::Text(corpus) => corpus.window(offset, ctx_len + 1),
                Self::BinIdx(corpus) => corpus.window(offset, ctx_len + 1)?,
            };

            inputs.extend_from_slice(&window[..ctx_len]);
            targets.extend_from_slice(&window[1..]);
        }

        Ok((inputs, targets))
    }
}

fn sample_offset(
    total_tokens: usize,
    ctx_len: usize,
    step: usize,
    row: usize,
    batch_size: usize,
    magic_prime: Option<u64>,
) -> usize {
    let max_offset = total_tokens - ctx_len - 1;
    if let Some(prime) = magic_prime.filter(|prime| *prime > 1) {
        let ii = 1_u128 + (step * batch_size + row) as u128;
        let factor = ((prime as f64) * ((5.0_f64.sqrt() - 1.0) / 2.0)).floor() as u128;
        let pos = ((factor * ii * ii * ii) % prime as u128) as usize * ctx_len;
        pos.min(max_offset)
    } else {
        ((step * batch_size + row) * ctx_len) % max_offset.max(1)
    }
}

#[derive(Debug)]
pub struct TextCorpus {
    tokens: Vec<i32>,
}

impl TextCorpus {
    fn load(path: &str, tokenizer: &WorldTokenizer) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read text corpus from {path}"))?;
        let tokens = tokenizer
            .encode(&text)
            .into_iter()
            .map(|token| token as i32)
            .collect::<Vec<_>>();

        Ok(Self { tokens })
    }

    fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    fn window(&self, offset: usize, len: usize) -> Vec<i32> {
        self.tokens[offset..offset + len].to_vec()
    }
}

#[derive(Debug)]
pub struct BinIdxCorpus {
    data: Mmap,
    dtype: BinIdxDType,
    total_tokens: usize,
}

impl BinIdxCorpus {
    fn load(path: &str) -> Result<Self> {
        let prefix = PathBuf::from(path);
        let idx_path = with_extension(&prefix, "idx");
        let bin_path = with_extension(&prefix, "bin");

        let idx_bytes = std::fs::read(&idx_path)
            .with_context(|| format!("failed to read binidx index file {}", idx_path.display()))?;
        parse_idx_header(&idx_bytes)?;

        let dtype = parse_dtype(&idx_bytes)?;
        let file = File::open(&bin_path)
            .with_context(|| format!("failed to open binidx data file {}", bin_path.display()))?;
        let data = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to memory-map {}", bin_path.display()))?;
        let total_tokens = data.len() / dtype.byte_len();

        Ok(Self {
            data,
            dtype,
            total_tokens,
        })
    }

    fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn window(&self, offset: usize, len: usize) -> Result<Vec<i32>> {
        if offset + len > self.total_tokens {
            bail!(
                "requested binidx window [{}..{}) exceeds corpus length {}",
                offset,
                offset + len,
                self.total_tokens
            );
        }

        let mut tokens = Vec::with_capacity(len);
        for index in offset..offset + len {
            tokens.push(self.read_token(index));
        }

        Ok(tokens)
    }

    fn read_token(&self, index: usize) -> i32 {
        let offset = index * self.dtype.byte_len();
        match self.dtype {
            BinIdxDType::U8 => self.data[offset] as i32,
            BinIdxDType::I8 => (self.data[offset] as i8) as i32,
            BinIdxDType::I16 => i16::from_le_bytes([self.data[offset], self.data[offset + 1]]) as i32,
            BinIdxDType::I32 => i32::from_le_bytes(
                self.data[offset..offset + 4]
                    .try_into()
                    .expect("exact slice for i32 token"),
            ),
            BinIdxDType::I64 => i64::from_le_bytes(
                self.data[offset..offset + 8]
                    .try_into()
                    .expect("exact slice for i64 token"),
            ) as i32,
            BinIdxDType::U16 => u16::from_le_bytes([self.data[offset], self.data[offset + 1]]) as i32,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BinIdxDType {
    U8,
    I8,
    I16,
    I32,
    I64,
    U16,
}

impl BinIdxDType {
    fn byte_len(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }
}

fn parse_idx_header(idx_bytes: &[u8]) -> Result<()> {
    if idx_bytes.len() < 26 {
        bail!("binidx header is too short");
    }

    let magic = &idx_bytes[..9];
    if magic != BINIDX_MAGIC {
        bail!("unexpected binidx magic header");
    }

    let version = u64::from_le_bytes(
        idx_bytes[9..17]
            .try_into()
            .expect("exact slice for binidx version"),
    );
    if version != 1 {
        bail!("unsupported binidx version {version}");
    }

    Ok(())
}

fn parse_dtype(idx_bytes: &[u8]) -> Result<BinIdxDType> {
    match idx_bytes[17] {
        1 => Ok(BinIdxDType::U8),
        2 => Ok(BinIdxDType::I8),
        3 => Ok(BinIdxDType::I16),
        4 => Ok(BinIdxDType::I32),
        5 => Ok(BinIdxDType::I64),
        8 => Ok(BinIdxDType::U16),
        code => bail!("unsupported binidx dtype code {code}"),
    }
}

fn with_extension(prefix: &Path, ext: &str) -> PathBuf {
    if prefix.extension().is_some() {
        prefix.to_path_buf()
    } else {
        prefix.with_extension(ext)
    }
}
