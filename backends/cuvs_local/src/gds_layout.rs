// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolves a fragment's uncompressed `FixedSizeList<f32, dim>` column straight to physical
//! `(file_path, byte_offset, byte_length)` ranges, bypassing `lance-encoding`'s normal decode
//! scheduler entirely -- the byte ranges the GDS read path (`cuda.rs`'s `DeviceTensor::
//! read_from_gds`, via `cuvsReadLargeFile`) needs.
//!
//! Scoped narrowly to Lance's 2.1 "structural" format, non-nullable `FullZipLayout` case only --
//! see `profiling/GDS_PORTING_PLAN.md` ("2.1 structural-format byte-range resolution research")
//! for the tracing this implements. A sibling module for the archived 2.0 "previous"/legacy
//! format exists only on `lance-cuvs`'s `archive/hand-rolled-gds-prototype-2026-08-26` branch --
//! not reachable from here, and this module deliberately does not attempt to support that format.
//!
//! Fails loudly (returns `Err`, never silently misreads) outside exactly this shape:
//! - single data file per fragment, no deletion vector
//! - file/column encoding is 2.1 "structural" (`ColumnInfo::is_structural()`), not legacy
//! - each page's structural layout is `PageLayout::FullZipLayout` with `bits_rep == 0 &&
//!   bits_def == 0` (non-nullable, not inside a list -- confirmed via encoder-side tracing to be
//!   the case that writes zero control-word/repetition-index overhead, making the on-disk bytes
//!   a plain contiguous float array) and `bits_per_value` matching `dimension * dtype_width`
//!   (uncompressed) -- `bits_per_value` is bits for one atomic FullZip "value", which for a
//!   FixedSizeList<f32,dim> column is the whole dim-float row, not one scalar (empirically
//!   confirmed via `gds_fragment_verify`: dim=2048 gave 65536, not 32 -- an earlier version of
//!   this check asserted plain dtype width and failed loudly with exactly that number)
//! - the target column is flat (not nested under a struct), so its position in
//!   `file_schema.fields` matches its `ColumnInfo` index
//!
//! Explicitly NOT supported yet (see `GDS_PORTING_PLAN.md`'s open-items list): nullable columns
//! (a real repetition-index buffer gets introduced, untraced), and cross-page byte-contiguity
//! coalescing (unlike the archived 2.0 module, this one does not attempt to merge adjacent pages'
//! reads -- whether `FullZipLayout` pages are contiguous across a fragment on disk was not
//! determined this session; per-page reads are always correct regardless, just not coalesced).

use lance::dataset::Dataset;
use lance::dataset::fragment::FileFragment;
use lance_core::{Error, Result};
use lance_encoding::decoder::{ColumnInfo, PageInfo};
use lance_encoding::format::pb21;
use lance_file::reader::FileReader;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;

/// One page's resolved value byte range, plus which rows (within the column's whole-fragment row
/// range) it covers.
#[derive(Debug, Clone, Copy)]
pub struct PagePlan {
    /// Cumulative row offset (within the column, across prior pages) this page starts at.
    pub row_start: u64,
    pub num_rows: u64,
    pub file_offset: u64,
    pub byte_len: u64,
}

/// Walks one page's structural encoding, asserting it's exactly a non-nullable, non-list
/// `FullZipLayout` with `expected_bits_per_value`, per this module's documented narrow scope.
/// Returns `(file_offset, byte_len)` for the page's values buffer.
///
/// Unlike the 2.0 "previous" format's `Buffer{buffer_index, buffer_type}` indirection, a
/// `FullZipLayout` page's values buffer is directly `page.buffer_offsets_and_sizes[0]` -- no
/// protobuf-driven buffer-index resolution needed (confirmed via `FullZipScheduler::try_new`,
/// see `GDS_PORTING_PLAN.md`).
fn resolve_page_full_zip(
    page: &PageInfo,
    expected_bits_per_value: u32,
) -> Result<(u64, u64)> {
    if !page.encoding.is_structural() {
        return Err(Error::not_supported(
            "page uses the 2.0 'previous' (legacy) encoding -- this GDS path only supports the \
             2.1 'structural' encoding",
        ));
    }
    let layout = page.encoding.as_structural();
    let full_zip = match layout.layout.as_ref() {
        Some(pb21::page_layout::Layout::FullZipLayout(full_zip)) => full_zip,
        other => {
            return Err(Error::not_supported(format!(
                "expected a FullZipLayout page (uncompressed, non-nullable, non-list \
                 FixedSizeList<f32> column), got {other:?}"
            )));
        }
    };
    if full_zip.bits_rep != 0 || full_zip.bits_def != 0 {
        return Err(Error::not_supported(format!(
            "expected no repetition/definition levels (non-nullable column, not inside a list), \
             got bits_rep={} bits_def={} -- nullable-column byte-range resolution is not yet \
             implemented (see GDS_PORTING_PLAN.md open items)",
            full_zip.bits_rep, full_zip.bits_def
        )));
    }
    let bits_per_value = match full_zip.details.as_ref() {
        Some(pb21::full_zip_layout::Details::BitsPerValue(bits)) => *bits,
        other => {
            return Err(Error::not_supported(format!(
                "expected fixed-width (BitsPerValue) FullZipLayout details, got {other:?}"
            )));
        }
    };
    // NOTE: `bits_per_value` here is bits for one *atomic FullZip "value"*, which for a
    // FixedSizeList<f32, dim> column is the whole dim-float row, not a single scalar -- e.g.
    // dim=2048 gives 2048*32=65536, not 32. Confirmed empirically via `gds_fragment_verify`
    // against a real dataset (the first version of this check asserted plain 32 and failed
    // loudly with exactly this number, which is what caught the bug) -- `expected_bits_per_value`
    // must already be `dimension * bytes_per_value * 8`, computed by the caller.
    if bits_per_value != expected_bits_per_value {
        return Err(Error::not_supported(format!(
            "expected {expected_bits_per_value} bits/value (dimension * bytes_per_value * 8), \
             got {bits_per_value} -- column may be compressed, which this GDS path does not \
             support"
        )));
    }
    // `value_compression` (a `CompressiveEncoding`) isn't explicitly asserted here to be
    // identity/uncompressed -- its exact "uncompressed" wire representation wasn't pinned down
    // during this module's research (see GDS_PORTING_PLAN.md). The values-buffer-size check in
    // `resolve_whole_column` below is a real, if indirect, safety net: if compression were
    // actually applied despite `bits_per_value` matching, the buffer size wouldn't match the
    // expected uncompressed size and that check would catch it.

    let (file_offset, _buffer_size) = page
        .buffer_offsets_and_sizes
        .first()
        .copied()
        .ok_or_else(|| Error::io("FullZipLayout page has no buffers"))?;
    Ok((file_offset, 0)) // byte_len computed by the caller, which knows num_rows/dimension/stride
}

/// Resolves every page of `column` (already verified to be an uncompressed, non-nullable
/// `FixedSizeList<f32, expected_dimension>` column) into a read plan covering the column's
/// entire row range, page by page.
///
/// `expected_dimension`/`expected_bytes_per_value` are asserted against every page, not just
/// trusted -- a mismatch means this reader's assumptions about the column's shape/encoding are
/// wrong, and continuing would silently produce wrong offsets rather than an error.
pub fn resolve_whole_column(
    column: &ColumnInfo,
    expected_dimension: u64,
    expected_bytes_per_value: u64,
) -> Result<Vec<PagePlan>> {
    // `FullZipLayout.bits_per_value` is bits for one atomic FullZip "value" -- the whole
    // dim-float row for a FixedSizeList<f32,dim> column, not one scalar -- see the note in
    // `resolve_page_full_zip`.
    let expected_bits_per_value = (expected_dimension * expected_bytes_per_value * 8) as u32;
    let stride = expected_dimension * expected_bytes_per_value;
    let mut out = Vec::with_capacity(column.page_infos.len());
    let mut row_cursor = 0u64;
    for (page_index, page) in column.page_infos.iter().enumerate() {
        if page.num_rows == 0 {
            continue;
        }
        let (file_offset, _) = resolve_page_full_zip(page, expected_bits_per_value)
            .map_err(|error| Error::io(format!("page {page_index}: {error}")))?;
        let byte_len = page.num_rows * stride;
        // The buffer's own recorded size is a real cross-check, not a formality -- see the
        // `value_compression` note in `resolve_page_full_zip` above.
        let (_, recorded_size) = page.buffer_offsets_and_sizes[0];
        if recorded_size != byte_len {
            return Err(Error::io(format!(
                "page {page_index}: values buffer size {recorded_size} does not match expected \
                 {byte_len} bytes for {} rows x {expected_dimension} dims x \
                 {expected_bytes_per_value} bytes/value -- encoding assumption is wrong",
                page.num_rows
            )));
        }
        out.push(PagePlan {
            row_start: row_cursor,
            num_rows: page.num_rows,
            file_offset,
            byte_len,
        });
        row_cursor += page.num_rows;
    }
    Ok(out)
}

/// Resolves `column` (an uncompressed `FixedSizeList<f32, expected_dimension>` column, asserted
/// at runtime) in `fragment` to a real local filesystem path plus per-page byte-range read plans,
/// without going through Lance's normal decode scheduler at all.
///
/// `dataset_root` is the plain local filesystem directory the dataset was opened from -- needed
/// because the GDS read path (`cuvsReadLargeFile`) does a raw file open, unlike Lance's own
/// `ObjectStore` abstraction which supports non-local backends a raw open can't reach; this GDS
/// path only ever makes sense for local block storage regardless (that's the entire premise of
/// GDS), so requiring a local root here is a scope match, not a new limitation.
///
/// Fails loudly (see module docs) on anything outside this module's narrow supported shape.
pub async fn resolve_fragment_column(
    dataset: &Dataset,
    dataset_root: &std::path::Path,
    fragment: &FileFragment,
    column: &str,
    expected_dimension: u64,
    expected_bytes_per_value: u64,
) -> Result<(std::path::PathBuf, Vec<PagePlan>)> {
    let metadata = fragment.metadata();
    if metadata.files.len() != 1 {
        return Err(Error::not_supported(format!(
            "fragment {} has {} data files; GDS path only supports exactly 1",
            metadata.id,
            metadata.files.len()
        )));
    }
    if metadata.deletion_file.is_some() {
        return Err(Error::not_supported(format!(
            "fragment {} has a deletion vector; GDS path does not support deletion-filtered \
             reads yet",
            metadata.id
        )));
    }
    let data_file = &metadata.files[0];
    if data_file.base_id.is_some() {
        return Err(Error::not_supported(
            "GDS path does not support non-default object-store bases",
        ));
    }
    if data_file.is_legacy_file() {
        return Err(Error::not_supported(
            "GDS path only supports 2.1 'structural' data files, got a legacy (pre-2.1) file",
        ));
    }

    let rel_path = dataset.data_dir().child(data_file.path.as_str());
    let file_size = dataset
        .object_store()
        .size(&rel_path)
        .await
        .map_err(|error| Error::io(format!("failed to stat {rel_path}: {error}")))?;
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(dataset.object_store()),
    );
    let file_scheduler = scheduler
        .open_file(&rel_path, &CachedFileSize::new(file_size))
        .await?;
    let file_metadata = FileReader::read_all_metadata(&file_scheduler).await?;

    if file_metadata.column_infos.len() != file_metadata.file_schema.fields.len() {
        return Err(Error::not_supported(
            "file schema has a nested/non-flat structure; GDS path only supports flat top-level \
             columns",
        ));
    }
    let column_index = file_metadata
        .file_schema
        .fields
        .iter()
        .position(|field| field.name == column)
        .ok_or_else(|| Error::invalid_input(format!("column '{column}' not found in file schema")))?;
    let column_info = &file_metadata.column_infos[column_index];
    if !column_info.is_structural() {
        return Err(Error::not_supported(
            "column's pages use the 2.0 'previous' (legacy) encoding, not 2.1 'structural' -- \
             GDS path only supports structural",
        ));
    }

    let plans = resolve_whole_column(column_info, expected_dimension, expected_bytes_per_value)?;
    let local_path = dataset_root.join("data").join(&data_file.path);
    Ok((local_path, plans))
}
