//! Compressed binary format for the shipped model bundle.
//!
//! Trees dominate the size (~500k nodes), so they are stored as the least information that
//! rebuilds them, split into streams of like data so the final brotli pass finds structure:
//!
//! * **topology** — one bit per node in preorder (`1` = split); child indexes are implied;
//! * **features** — `u16` stat column per split;
//! * **thresholds** — `u8` per split, an index into that ensemble's cut table for the
//!   column. Every threshold is one of at most 256 training bin boundaries per column, and
//!   the table holds the exact `f32` values, so splits are reproduced bit for bit;
//! * **leaves** — `i16` per leaf with one `f32` scale per tree (`max |value| / 32767`), an
//!   error of at most half a step: far below 0.1 MMR once the learning rate applies.
//!
//! Layout: [`MAGIC`] then brotli(payload). The payload is the header JSON (length-prefixed;
//! stat names, layouts, scales, coaching and mechanic tables) followed by the ensembles in a
//! fixed order (match absolute, match within, window absolute, window within, then one per
//! plays-like group in header order). Integers and floats are little-endian.

use serde::{Deserialize, Serialize};

use crate::coaching::CoachingTables;
use crate::mechanics::MechanicTables;
use crate::model::{FrozenEnsemble, FrozenNode, PlaysLikeModel, TabularLayout, TabularSkillModel};
use crate::plays_like::{PlaysLikeGroup, PlaysLikeGroups};

/// File signature and format version.
const MAGIC: &[u8; 8] = b"RLSKILL5";

/// Largest quantised leaf magnitude.
#[cfg(feature = "encode")]
const LEAF_STEPS: f32 = 32767.0;

/// Everything the app needs, in one artifact.
#[derive(Debug, Clone)]
pub struct SkillModelBundle {
    /// Scores whole matches: the verdict and the cards.
    pub match_model: TabularSkillModel,
    /// Scores fixed windows of live play: the timeline.
    pub window_model: TabularSkillModel,
    /// Score the rank each player's own stats play like, one model per stat group: the
    /// smurf flag.
    pub plays_like_groups: PlaysLikeGroups,
    /// Window length the window model was trained on, in seconds of live play.
    pub window_seconds: f32,
    /// Multiplier applied to a player's window-to-window form on the timeline so a typical
    /// swing reads as about one tier. Display only; never used for the verdict.
    pub timeline_emphasis: f32,
    /// Per-tier reference stats for the "next rank" roast, per playlist size.
    pub coaching: CoachingTables,
    /// Per-tier mechanic rates: which mechanics a player does far more than their lobby.
    pub mechanics: MechanicTables,
}

/// Why a bundle could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleDecodeError {
    /// Not a bundle, or a different format version.
    BadMagic,
    /// The data ended early.
    Truncated,
    /// The compressed payload is corrupt.
    Decompression,
    /// The header JSON did not parse.
    Header(String),
    /// A tree references a split or cut value that is not there.
    BadTree,
}

impl core::fmt::Display for BundleDecodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(formatter, "not a skill model bundle (bad signature)"),
            Self::Truncated => write!(formatter, "skill model bundle is truncated"),
            Self::Decompression => write!(formatter, "skill model bundle is corrupt"),
            Self::Header(message) => write!(formatter, "bad bundle header: {message}"),
            Self::BadTree => write!(formatter, "bundle contains a malformed tree"),
        }
    }
}

impl std::error::Error for BundleDecodeError {}

/// Why a bundle could not be encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleEncodeError {
    /// A column has more than 256 distinct thresholds, or a count overflows its field.
    UnsupportedTree,
    /// The header could not be serialised.
    Header(String),
    /// The compressor failed.
    Compression(String),
}

impl core::fmt::Display for BundleEncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedTree => write!(formatter, "tree cannot be encoded"),
            Self::Header(message) => write!(formatter, "header cannot be encoded: {message}"),
            Self::Compression(message) => write!(formatter, "compression failed: {message}"),
        }
    }
}

impl std::error::Error for BundleEncodeError {}

/// The non-tree half of a [`TabularSkillModel`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelHeader {
    absolute_layout: TabularLayout,
    within_layout: TabularLayout,
    deviation_scale: f32,
}

/// The non-tree half of a [`PlaysLikeGroup`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlaysLikeGroupHeader {
    name: String,
    weight: f32,
    layout: TabularLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BundleHeader {
    stat_names: Vec<String>,
    match_model: ModelHeader,
    window_model: ModelHeader,
    plays_like_groups: Vec<PlaysLikeGroupHeader>,
    window_seconds: f32,
    timeline_emphasis: f32,
    coaching: CoachingTables,
    mechanics: MechanicTables,
}

impl ModelHeader {
    #[cfg(feature = "encode")]
    fn of(model: &TabularSkillModel) -> Self {
        Self {
            absolute_layout: model.absolute_layout.clone(),
            within_layout: model.within_layout.clone(),
            deviation_scale: model.deviation_scale,
        }
    }

    fn into_model(
        self,
        stat_names: Vec<String>,
        absolute: FrozenEnsemble,
        within: FrozenEnsemble,
    ) -> TabularSkillModel {
        TabularSkillModel {
            stat_names,
            absolute_layout: self.absolute_layout,
            absolute,
            within_layout: self.within_layout,
            within,
            deviation_scale: self.deviation_scale,
        }
    }
}

// ----------------------------------------------------------------------------------------
// Encoding
// ----------------------------------------------------------------------------------------

/// Little-endian writer for the payload.
#[cfg(feature = "encode")]
#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

#[cfg(feature = "encode")]
impl Writer {
    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn count(&mut self, value: usize) -> Result<(), BundleEncodeError> {
        self.u32(
            u32::try_from(value)
                .ok()
                .ok_or(BundleEncodeError::UnsupportedTree)?,
        );
        Ok(())
    }
}

/// One ensemble split into its streams.
#[cfg(feature = "encode")]
#[derive(Default)]
struct EnsembleStreams {
    topology: Vec<bool>,
    features: Vec<u16>,
    thresholds: Vec<f32>,
    leaf_scales: Vec<f32>,
    leaves: Vec<i16>,
}

#[cfg(feature = "encode")]
impl EnsembleStreams {
    /// Walks one tree in preorder, recording its shape and contents.
    fn add_tree(&mut self, tree: &[FrozenNode]) -> Result<(), BundleEncodeError> {
        let largest_leaf = tree
            .iter()
            .filter_map(|node| match node {
                FrozenNode::Leaf { value } => Some(value.abs()),
                FrozenNode::Split { .. } => None,
            })
            .fold(0.0_f32, f32::max);
        let scale = if largest_leaf > 0.0 {
            largest_leaf / LEAF_STEPS
        } else {
            1.0
        };
        self.leaf_scales.push(scale);

        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            match tree.get(index) {
                Some(FrozenNode::Leaf { value }) => {
                    self.topology.push(false);
                    self.leaves.push((value / scale).round() as i16);
                }
                Some(FrozenNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                }) => {
                    self.topology.push(true);
                    self.features.push(
                        u16::try_from(*feature)
                            .ok()
                            .ok_or(BundleEncodeError::UnsupportedTree)?,
                    );
                    self.thresholds.push(*threshold);
                    // Pushed right first so the left subtree is written first (preorder).
                    stack.push(*right);
                    stack.push(*left);
                }
                None => return Err(BundleEncodeError::UnsupportedTree),
            }
        }
        Ok(())
    }

    fn write(self, ensemble: &FrozenEnsemble, out: &mut Writer) -> Result<(), BundleEncodeError> {
        out.f32(ensemble.base_prediction);
        out.f32(ensemble.learning_rate);
        out.count(ensemble.trees.len())?;

        // Cut tables: each used column's distinct thresholds, sorted.
        let mut columns: Vec<u16> = self.features.clone();
        columns.sort_unstable();
        columns.dedup();
        let mut cut_tables: Vec<CutTable> = columns
            .iter()
            .map(|&column| CutTable {
                column,
                values: Vec::new(),
            })
            .collect();
        for (column, threshold) in self.features.iter().zip(&self.thresholds) {
            if let Ok(position) = columns.binary_search(column)
                && let Some(table) = cut_tables.get_mut(position)
            {
                table.values.push(*threshold);
            }
        }
        out.count(cut_tables.len())?;
        for table in &mut cut_tables {
            table.values.sort_by(f32::total_cmp);
            table.values.dedup_by(|a, b| a.to_bits() == b.to_bits());
            if table.values.len() > 256 {
                return Err(BundleEncodeError::UnsupportedTree);
            }
            out.u16(table.column);
            out.u16(
                u16::try_from(table.values.len())
                    .ok()
                    .ok_or(BundleEncodeError::UnsupportedTree)?,
            );
            for value in &table.values {
                out.f32(*value);
            }
        }

        // Topology bits, packed.
        out.count(self.topology.len())?;
        for chunk in self.topology.chunks(8) {
            let byte = chunk
                .iter()
                .enumerate()
                .fold(0u8, |byte, (bit, &split)| byte | (u8::from(split) << bit));
            out.bytes.push(byte);
        }
        // Split columns, then threshold indexes.
        out.count(self.features.len())?;
        for column in &self.features {
            out.u16(*column);
        }
        for (column, threshold) in self.features.iter().zip(&self.thresholds) {
            let table = columns
                .binary_search(column)
                .ok()
                .and_then(|position| cut_tables.get(position))
                .ok_or(BundleEncodeError::UnsupportedTree)?;
            let index = table
                .values
                .iter()
                .position(|value| value.to_bits() == threshold.to_bits())
                .ok_or(BundleEncodeError::UnsupportedTree)?;
            out.bytes.push(
                u8::try_from(index)
                    .ok()
                    .ok_or(BundleEncodeError::UnsupportedTree)?,
            );
        }
        // Leaves: per-tree scales, then quantised values.
        for scale in &self.leaf_scales {
            out.f32(*scale);
        }
        out.count(self.leaves.len())?;
        for leaf in &self.leaves {
            out.bytes.extend_from_slice(&leaf.to_le_bytes());
        }
        Ok(())
    }
}

/// One column's distinct thresholds.
#[cfg(feature = "encode")]
struct CutTable {
    column: u16,
    values: Vec<f32>,
}

#[cfg(feature = "encode")]
fn write_ensemble(ensemble: &FrozenEnsemble, out: &mut Writer) -> Result<(), BundleEncodeError> {
    let mut streams = EnsembleStreams::default();
    for tree in &ensemble.trees {
        streams.add_tree(tree)?;
    }
    streams.write(ensemble, out)
}

// ----------------------------------------------------------------------------------------
// Decoding
// ----------------------------------------------------------------------------------------

/// Reads little-endian values off the front of a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn take(&mut self, count: usize) -> Result<&'a [u8], BundleDecodeError> {
        if self.bytes.len() < count {
            return Err(BundleDecodeError::Truncated);
        }
        let (head, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], BundleDecodeError> {
        let mut array = [0u8; N];
        array.copy_from_slice(self.take(N)?);
        Ok(array)
    }

    fn u16(&mut self) -> Result<u16, BundleDecodeError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, BundleDecodeError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn count(&mut self) -> Result<usize, BundleDecodeError> {
        Ok(self.u32()? as usize)
    }

    fn f32(&mut self) -> Result<f32, BundleDecodeError> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    fn ensemble(&mut self) -> Result<FrozenEnsemble, BundleDecodeError> {
        let base_prediction = self.f32()?;
        let learning_rate = self.f32()?;
        let tree_count = self.count()?;

        let table_count = self.count()?;
        let mut cut_tables: Vec<DecodedCutTable> = Vec::with_capacity(table_count);
        for _ in 0..table_count {
            let column = self.u16()?;
            let length = usize::from(self.u16()?);
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(self.f32()?);
            }
            cut_tables.push(DecodedCutTable { column, values });
        }

        let node_count = self.count()?;
        let topology = self.take(node_count.div_ceil(8))?;
        let split_count = self.count()?;
        let mut features = Vec::with_capacity(split_count);
        for _ in 0..split_count {
            features.push(self.u16()?);
        }
        let threshold_indexes = self.take(split_count)?;
        let mut leaf_scales = Vec::with_capacity(tree_count);
        for _ in 0..tree_count {
            leaf_scales.push(self.f32()?);
        }
        let leaf_count = self.count()?;
        let leaves = self.take(leaf_count * 2)?;

        let mut cursor = StreamCursor {
            topology,
            node: 0,
            node_count,
            split: 0,
            leaf: 0,
        };
        let mut trees = Vec::with_capacity(tree_count);
        for scale in leaf_scales {
            trees.push(decode_tree(
                &mut cursor,
                &DecodeStreams {
                    features: &features,
                    threshold_indexes,
                    cut_tables: &cut_tables,
                    leaves,
                    scale,
                },
            )?);
        }
        Ok(FrozenEnsemble {
            base_prediction,
            learning_rate,
            trees,
        })
    }
}

/// One column's cut table, as read.
struct DecodedCutTable {
    column: u16,
    values: Vec<f32>,
}

/// Read-only streams shared by every tree of an ensemble.
struct DecodeStreams<'a> {
    features: &'a [u16],
    threshold_indexes: &'a [u8],
    cut_tables: &'a [DecodedCutTable],
    leaves: &'a [u8],
    /// This tree's leaf scale.
    scale: f32,
}

/// Positions in the streams, advanced as trees are rebuilt.
struct StreamCursor<'a> {
    topology: &'a [u8],
    node: usize,
    node_count: usize,
    split: usize,
    leaf: usize,
}

impl StreamCursor<'_> {
    fn next_is_split(&mut self) -> Result<bool, BundleDecodeError> {
        if self.node >= self.node_count {
            return Err(BundleDecodeError::BadTree);
        }
        let byte = self
            .topology
            .get(self.node / 8)
            .ok_or(BundleDecodeError::BadTree)?;
        let split = (byte >> (self.node % 8)) & 1 == 1;
        self.node += 1;
        Ok(split)
    }
}

/// Rebuilds one tree from its preorder stream. Children of a split are allocated as an
/// adjacent pair, the same layout the trainer produces.
fn decode_tree(
    cursor: &mut StreamCursor<'_>,
    streams: &DecodeStreams<'_>,
) -> Result<Vec<FrozenNode>, BundleDecodeError> {
    let mut nodes = vec![FrozenNode::Leaf { value: 0.0 }];
    let mut stack = vec![0usize];
    while let Some(index) = stack.pop() {
        let node = if cursor.next_is_split()? {
            let feature = *streams
                .features
                .get(cursor.split)
                .ok_or(BundleDecodeError::BadTree)?;
            let threshold_index = *streams
                .threshold_indexes
                .get(cursor.split)
                .ok_or(BundleDecodeError::BadTree)?;
            cursor.split += 1;
            let threshold = streams
                .cut_tables
                .iter()
                .find(|table| table.column == feature)
                .and_then(|table| table.values.get(usize::from(threshold_index)))
                .copied()
                .ok_or(BundleDecodeError::BadTree)?;
            let left = nodes.len();
            nodes.push(FrozenNode::Leaf { value: 0.0 });
            nodes.push(FrozenNode::Leaf { value: 0.0 });
            stack.push(left + 1);
            stack.push(left);
            FrozenNode::Split {
                feature: usize::from(feature),
                threshold,
                left,
                right: left + 1,
            }
        } else {
            let bytes = streams
                .leaves
                .get(cursor.leaf * 2..cursor.leaf * 2 + 2)
                .ok_or(BundleDecodeError::BadTree)?;
            cursor.leaf += 1;
            let quantised = i16::from_le_bytes([
                bytes.first().copied().unwrap_or(0),
                bytes.get(1).copied().unwrap_or(0),
            ]);
            FrozenNode::Leaf {
                value: f32::from(quantised) * streams.scale,
            }
        };
        if let Some(slot) = nodes.get_mut(index) {
            *slot = node;
        }
    }
    Ok(nodes)
}

impl SkillModelBundle {
    /// Encodes and compresses the bundle.
    ///
    /// # Errors
    ///
    /// [`BundleEncodeError`] when a tree does not fit the format.
    #[cfg(feature = "encode")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, BundleEncodeError> {
        let header = BundleHeader {
            stat_names: self.match_model.stat_names.clone(),
            match_model: ModelHeader::of(&self.match_model),
            window_model: ModelHeader::of(&self.window_model),
            plays_like_groups: self
                .plays_like_groups
                .groups
                .iter()
                .map(|group| PlaysLikeGroupHeader {
                    name: group.name.clone(),
                    weight: group.weight,
                    layout: group.model.layout.clone(),
                })
                .collect(),
            window_seconds: self.window_seconds,
            timeline_emphasis: self.timeline_emphasis,
            coaching: self.coaching.clone(),
            mechanics: self.mechanics.clone(),
        };
        let header_json = serde_json::to_vec(&header)
            .map_err(|error| BundleEncodeError::Header(error.to_string()))?;
        let mut payload = Writer::default();
        payload.count(header_json.len())?;
        payload.bytes.extend_from_slice(&header_json);
        for ensemble in [
            &self.match_model.absolute,
            &self.match_model.within,
            &self.window_model.absolute,
            &self.window_model.within,
        ]
        .into_iter()
        .chain(
            self.plays_like_groups
                .groups
                .iter()
                .map(|group| &group.model.ensemble),
        ) {
            write_ensemble(ensemble, &mut payload)?;
        }

        let mut out = MAGIC.to_vec();
        let parameters = brotli::enc::BrotliEncoderParams {
            quality: 11,
            lgwin: 24,
            ..brotli::enc::BrotliEncoderParams::default()
        };
        brotli::BrotliCompress(&mut payload.bytes.as_slice(), &mut out, &parameters)
            .map_err(|error| BundleEncodeError::Compression(error.to_string()))?;
        Ok(out)
    }

    /// Decodes a bundle written by `to_bytes`.
    ///
    /// # Errors
    ///
    /// Any [`BundleDecodeError`]; a bundle trained on a different stat list decodes fine
    /// and is caught by [`TabularSkillModel::matches_current_stats`] instead.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BundleDecodeError> {
        let Some(compressed) = bytes.strip_prefix(MAGIC.as_slice()) else {
            return Err(BundleDecodeError::BadMagic);
        };
        let mut payload = Vec::new();
        brotli_decompressor::BrotliDecompress(&mut &*compressed, &mut payload)
            .map_err(|_error| BundleDecodeError::Decompression)?;

        let mut reader = Reader { bytes: &payload };
        let header_length = reader.count()?;
        let header: BundleHeader = serde_json::from_slice(reader.take(header_length)?)
            .map_err(|error| BundleDecodeError::Header(error.to_string()))?;
        let match_absolute = reader.ensemble()?;
        let match_within = reader.ensemble()?;
        let window_absolute = reader.ensemble()?;
        let window_within = reader.ensemble()?;
        let mut groups = Vec::with_capacity(header.plays_like_groups.len());
        for group in header.plays_like_groups {
            groups.push(PlaysLikeGroup {
                name: group.name,
                weight: group.weight,
                model: PlaysLikeModel {
                    layout: group.layout,
                    ensemble: reader.ensemble()?,
                },
            });
        }
        Ok(Self {
            match_model: header.match_model.into_model(
                header.stat_names.clone(),
                match_absolute,
                match_within,
            ),
            window_model: header.window_model.into_model(
                header.stat_names,
                window_absolute,
                window_within,
            ),
            plays_like_groups: PlaysLikeGroups { groups },
            window_seconds: header.window_seconds,
            timeline_emphasis: header.timeline_emphasis,
            coaching: header.coaching,
            mechanics: header.mechanics,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "encode")]
mod tests {
    use super::*;

    fn tree() -> Vec<FrozenNode> {
        // A split whose left child is itself a split: exercises preorder rebuilding.
        vec![
            FrozenNode::Split {
                feature: 2,
                threshold: 0.5,
                left: 1,
                right: 2,
            },
            FrozenNode::Split {
                feature: 0,
                threshold: -1.25,
                left: 3,
                right: 4,
            },
            FrozenNode::Leaf { value: 4.0 },
            FrozenNode::Leaf { value: -3.0 },
            FrozenNode::Leaf { value: 1.5 },
        ]
    }

    fn model(base: f32) -> TabularSkillModel {
        let layout = TabularLayout {
            own: true,
            deviation: false,
            lobby_mean: false,
            team_context: false,
            stats: vec![0, 1, 2],
        };
        TabularSkillModel {
            stat_names: vec!["a".into(), "b".into(), "c".into()],
            absolute_layout: layout.clone(),
            absolute: FrozenEnsemble {
                base_prediction: base,
                learning_rate: 0.1,
                trees: vec![tree(), tree()],
            },
            within_layout: layout,
            within: FrozenEnsemble {
                base_prediction: 0.0,
                learning_rate: 1.0,
                trees: vec![tree()],
            },
            deviation_scale: 1.1,
        }
    }

    fn bundle() -> SkillModelBundle {
        SkillModelBundle {
            match_model: model(900.0),
            window_model: model(800.0),
            plays_like_groups: PlaysLikeGroups {
                groups: vec![PlaysLikeGroup {
                    name: "aerial".into(),
                    weight: 2.0,
                    model: PlaysLikeModel {
                        layout: model(0.0).absolute_layout,
                        ensemble: FrozenEnsemble {
                            base_prediction: 700.0,
                            learning_rate: 0.2,
                            trees: vec![tree()],
                        },
                    },
                }],
            },
            window_seconds: 60.0,
            timeline_emphasis: 5.0,
            coaching: CoachingTables::default(),
            mechanics: MechanicTables::default(),
        }
    }

    #[test]
    fn round_trip_preserves_predictions() {
        let original = bundle();
        let bytes = original.to_bytes().expect("encodes");
        let decoded = SkillModelBundle::from_bytes(&bytes).expect("decodes");
        for row in [[-2.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.0, 0.0, 0.9]] {
            for (before, after) in [
                (
                    &original.match_model.absolute,
                    &decoded.match_model.absolute,
                ),
                (&original.window_model.within, &decoded.window_model.within),
                (
                    &original.plays_like_groups.groups[0].model.ensemble,
                    &decoded.plays_like_groups.groups[0].model.ensemble,
                ),
            ] {
                let difference = (before.predict(&row) - after.predict(&row)).abs();
                assert!(difference < 1e-3, "prediction moved by {difference}");
            }
        }
        assert!((decoded.timeline_emphasis - 5.0).abs() < f32::EPSILON);
        assert!((decoded.match_model.deviation_scale - 1.1).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_foreign_and_truncated_data() {
        assert_eq!(
            SkillModelBundle::from_bytes(b"not a bundle").err(),
            Some(BundleDecodeError::BadMagic)
        );
        let bytes = bundle().to_bytes().expect("encodes");
        assert!(SkillModelBundle::from_bytes(&bytes[..bytes.len() - 3]).is_err());
    }
}
