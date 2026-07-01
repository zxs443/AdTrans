import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from pymatgen.core import Composition
from .dataset import MaterialsDataset
from data.structure_placement import (
    uses_structure_input,
    structure_in_decoder,
    structure_in_encoder,
    encoder_seq_len,
    encoder_structure_offset,
    get_structure_placement,
    structure_dual_token_split,
    structure_input_dim,
    structure_per_token_dim,
    n_structure_tokens,
    append_decoder_structure_token_names,
)
from utils.config import DATASET_CONFIG
from utils.token_ranking import token_name_from_decoder_column
from utils.formula_utils import _normalize_formula, _count_unique_elements_in_formula

# t-t mode: train/val ratio inside the train.csv pool.
TT_POOL_TRAIN_RATIO = 0.875
TT_POOL_VAL_RATIO = 0.125

N_ELEMENTS = 6
DESCRIPTOR_ROOT = 'descriptors'
MERGE_KEY = 'battery_id'
METADATA_COLUMNS = ['average_voltage', 'formula_discharge', 'battery_id']
STRUCTURE_COLUMN_PREFIX = 'chg_struct_emb'
NORMALIZATION_EPS = 1e-8

def _safe_standardize(data, mean, std, eps=None):
    """Z-score; zero-variance columns → 0 after scaling."""
    if eps is None:
        eps = NORMALIZATION_EPS

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    scale = np.where(std < eps, 1.0, std)
    normed = (data - mean) / scale

    zero_var_mask = std < eps
    if np.any(zero_var_mask):
        normed = np.where(zero_var_mask, 0.0, normed)

    return normed.astype(np.float32)

def _report_zero_variance_features(std, feature_scope):
    eps = NORMALIZATION_EPS
    zero_var_count = int(np.sum(std < eps))
    if zero_var_count > 0:
        print(
            f"Warning: {zero_var_count} zero-variance feature(s) detected in {feature_scope}. "
            "These features are set to 0 after standardization."
        )

def _sort_structure_columns(columns):
    prefix = STRUCTURE_COLUMN_PREFIX

    def _key(col):
        suffix = col[len(prefix) + 1 :] if col.startswith(prefix + '_') else col
        try:
            return int(suffix)
        except (TypeError, ValueError):
            return col

    return sorted(columns, key=_key)


def _validate_structure_columns(structure_columns):
    prefix = STRUCTURE_COLUMN_PREFIX
    invalid_columns = [col for col in structure_columns if not col.startswith(prefix)]
    if invalid_columns:
        preview = invalid_columns[:5]
        raise ValueError(
            f"Structure columns must start with '{prefix}'. Invalid examples: {preview}. "
            "Check structure_base/chgnet CSV files, or set structure_placement to 'none'."
        )
    if structure_dual_token_split():
        expected = structure_input_dim()
        per_token = structure_per_token_dim()
        if len(structure_columns) != expected:
            raise ValueError(
                f"structure_dual_token_split expects {expected} flat structure columns "
                f"(2 x {per_token}-d tokens), got {len(structure_columns)}."
            )
    else:
        expected = structure_input_dim()
        if len(structure_columns) != expected:
            raise ValueError(
                f"Expected {expected} structure columns, got {len(structure_columns)}."
            )


def _reshape_structure_features(structure_features: np.ndarray) -> np.ndarray:
    if structure_features is None:
        return None
    flat_dim = structure_input_dim()
    if structure_features.shape[1] != flat_dim:
        raise ValueError(
            f"Structure features width {structure_features.shape[1]} != expected {flat_dim}."
        )
    if not structure_dual_token_split():
        return structure_features
    per_token = structure_per_token_dim()
    n_tokens = n_structure_tokens()
    return structure_features.reshape(structure_features.shape[0], n_tokens, per_token)


def _standardize_structure_array(structure_data, structure_mean, structure_std):
    if structure_data is None:
        return None
    if structure_data.ndim == 3:
        out = np.empty_like(structure_data, dtype=np.float32)
        for token_idx in range(structure_data.shape[1]):
            out[:, token_idx, :] = _safe_standardize(
                structure_data[:, token_idx, :],
                structure_mean[token_idx],
                structure_std[token_idx],
            )
        return out
    return _safe_standardize(structure_data, structure_mean, structure_std)


def _structure_normalization_stats(structure_data):
    if structure_data is None:
        return None, None
    if structure_data.ndim == 3:
        structure_mean = np.mean(structure_data, axis=0)
        structure_std = np.std(structure_data, axis=0)
    else:
        structure_mean = np.mean(structure_data, axis=0)
        structure_std = np.std(structure_data, axis=0)
    return structure_mean, structure_std

def get_decoder_token_names(decoder_feature_columns, decoder_token_dim):
    extracted_token_names = []
    num_features = len(decoder_feature_columns)
    if num_features % decoder_token_dim != 0:
        raise ValueError(
            f"Error: The number of property decoder features {num_features} is not divisible "
            f"by the dimension per token {decoder_token_dim}."
        )
    for i in range(0, num_features, decoder_token_dim):
        first_col_in_group = decoder_feature_columns[i]
        extracted_token_names.append(token_name_from_decoder_column(first_col_in_group))
    if structure_in_decoder():
        extracted_token_names = append_decoder_structure_token_names(extracted_token_names)
    return extracted_token_names

def get_descriptor_root(base_path="."):
    root = DESCRIPTOR_ROOT
    if os.path.isabs(root):
        resolved = root
    else:
        resolved = os.path.normpath(os.path.join(base_path, root))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"descriptor_root does not exist: {resolved} "
            f"(expected {root!r} relative to base_path={base_path!r})."
        )
    return resolved


def _read_descriptor_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Descriptor file not found: {path}")
    df = pd.read_csv(path)
    if 'formula_discharge' not in df.columns:
        if 'formula' in df.columns:
            df = df.copy()
            df['formula_discharge'] = df['formula']
        else:
            raise ValueError(f"Descriptor file must contain 'formula_discharge' or 'formula': {path}")
    if 'battery_id' not in df.columns:
        raise ValueError(f"Descriptor file must contain 'battery_id': {path}")
    df = df.copy()
    df['formula_discharge'] = df['formula_discharge'].apply(_normalize_formula)
    df['battery_id'] = df['battery_id'].astype(str)
    return df


def _descriptor_component_dirs(encoder_descriptor_type, decoder_descriptor_type):
    components = [
        ('element_base', encoder_descriptor_type),
        ('property_base', decoder_descriptor_type),
    ]
    if uses_structure_input():
        structure_name = DATASET_CONFIG.get('structure_descriptor_name', 'chgnet')
        components.append(('structure_base', structure_name))
    return components


def _component_split_path(descriptor_root, base_dir, descriptor_name, split_name):
    return _split_csv_path(descriptor_root, base_dir, descriptor_name, split_name)


def _split_csv_path(descriptor_root, base_dir, descriptor_name, split_name):
    return os.path.join(descriptor_root, base_dir, descriptor_name, f'{split_name}.csv')


def detect_descriptor_split_mode(base_path, encoder_descriptor_type, decoder_descriptor_type):
    """
    Infer split mode from descriptor CSV files:
      - train + val + test  -> 't-v-t'
      - train + test (no val) -> 't-t'
    """
    descriptor_root = get_descriptor_root(base_path)
    components = _descriptor_component_dirs(encoder_descriptor_type, decoder_descriptor_type)

    def _all_have(split_name):
        return all(
            os.path.isfile(_component_split_path(descriptor_root, base_dir, desc_name, split_name))
            for base_dir, desc_name in components
        )

    def _any_have(split_name):
        return any(
            os.path.isfile(_component_split_path(descriptor_root, base_dir, desc_name, split_name))
            for base_dir, desc_name in components
        )

    has_train = _all_have('train')
    has_val = _all_have('val')
    has_test = _all_have('test')
    val_partial = _any_have('val') and not has_val

    if val_partial:
        raise ValueError(
            f"Inconsistent descriptor splits under {descriptor_root}: val.csv exists for some "
            "descriptor types but not all. For t-v-t, every component needs train/val/test; "
            "for t-t, val.csv must be absent everywhere."
        )

    if has_train and has_val and has_test:
        return 't-v-t'
    if has_train and has_test and not has_val:
        return 't-t'

    raise ValueError(
        f"Invalid descriptor split files under {descriptor_root}. "
        "Expected either train+val+test (t-v-t) or train+test without val (t-t). "
        f"Found train={has_train}, val={has_val}, test={has_test}."
    )


def _descriptor_paths(descriptor_root, encoder_descriptor_type, decoder_descriptor_type, split_name):
    element_path = os.path.join(descriptor_root, 'element_base', encoder_descriptor_type, f'{split_name}.csv')
    property_path = os.path.join(descriptor_root, 'property_base', decoder_descriptor_type, f'{split_name}.csv')
    structure_path = None
    if uses_structure_input():
        structure_name = DATASET_CONFIG.get('structure_descriptor_name', 'chgnet')
        structure_path = os.path.join(descriptor_root, 'structure_base', structure_name, f'{split_name}.csv')
    return element_path, property_path, structure_path


def _align_descriptor_tables(element_df, property_df, structure_df, n_elements):
    meta_cols = set(METADATA_COLUMNS)

    ordered_ids = element_df[MERGE_KEY].astype(str).tolist()
    property_indexed = property_df.set_index(property_df[MERGE_KEY].astype(str))
    missing_in_property = [bid for bid in ordered_ids if bid not in property_indexed.index]
    if missing_in_property:
        raise ValueError(
            f"Property descriptors missing {len(missing_in_property)} battery_id(s) present in element data. "
            f"Examples: {missing_in_property[:5]}"
        )

    encoder_cols = [c for c in element_df.columns if c not in meta_cols]
    property_cols = [c for c in property_df.columns if c not in meta_cols]
    element_indexed = element_df.set_index(element_df[MERGE_KEY].astype(str))

    encoder_features_df = element_indexed.loc[ordered_ids, encoder_cols]
    property_features_df = property_indexed.loc[ordered_ids, property_cols]

    structure_features_df = None
    structure_cols = []
    if structure_df is not None:
        structure_indexed = structure_df.set_index(structure_df[MERGE_KEY].astype(str))
        missing_in_structure = [bid for bid in ordered_ids if bid not in structure_indexed.index]
        if missing_in_structure:
            raise ValueError(
                f"Structure descriptors missing {len(missing_in_structure)} battery_id(s). "
                f"Examples: {missing_in_structure[:5]}"
            )
        structure_cols = _sort_structure_columns(
            [c for c in structure_df.columns if c not in meta_cols]
        )
        _validate_structure_columns(structure_cols)
        structure_features_df = structure_indexed.loc[ordered_ids, structure_cols]

    if encoder_features_df.isna().any().any():
        raise ValueError("NaN detected in aligned encoder descriptor features.")
    if property_features_df.isna().any().any():
        raise ValueError("NaN detected in aligned property descriptor features.")
    if structure_features_df is not None and structure_features_df.isna().any().any():
        raise ValueError("NaN detected in aligned structure descriptor features.")

    property_features = property_features_df.values
    structure_features = None
    if structure_features_df is not None:
        structure_features = _reshape_structure_features(structure_features_df.values)

    decoder_data = property_features
    decoder_feature_columns = property_cols

    targets = element_indexed.loc[ordered_ids, 'average_voltage'].values
    formulas = element_indexed.loc[ordered_ids, 'formula_discharge'].values
    battery_ids = np.array(ordered_ids, dtype=object)

    features_per_element = DATASET_CONFIG['encoder_token_dim']
    expected_encoder_cols = n_elements * features_per_element
    if encoder_features_df.shape[1] != expected_encoder_cols:
        raise ValueError(
            f"Encoder feature columns {encoder_features_df.shape[1]} do not match expected "
            f"({expected_encoder_cols} = {n_elements} * {features_per_element})."
        )

    return (
        encoder_features_df.values,
        decoder_data,
        structure_features,
        targets,
        formulas,
        battery_ids,
        decoder_feature_columns,
    )


def _load_descriptor_from_splits(base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, split_names):
    descriptor_root = get_descriptor_root(base_path)
    element_parts = []
    property_parts = []
    structure_parts = []

    for split_name in split_names:
        element_path, property_path, structure_path = _descriptor_paths(
            descriptor_root, encoder_descriptor_type, decoder_descriptor_type, split_name
        )
        element_parts.append(_read_descriptor_csv(element_path))
        property_parts.append(_read_descriptor_csv(property_path))
        if structure_path is not None:
            structure_parts.append(_read_descriptor_csv(structure_path))

    element_df = pd.concat(element_parts, ignore_index=True)
    property_df = pd.concat(property_parts, ignore_index=True)
    structure_df = pd.concat(structure_parts, ignore_index=True) if structure_parts else None
    return _align_descriptor_tables(element_df, property_df, structure_df, n_elements)


def load_descriptor_split(base_path, split_name, encoder_descriptor_type, decoder_descriptor_type, n_elements):
    """Load one split CSV set (train / val / test)."""
    return _load_descriptor_from_splits(
        base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, [split_name]
    )


def load_all_descriptor_splits(base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements):
    """Load train, val, and test splits for t-v-t mode."""
    split_data = {}
    for split_name in ('train', 'val', 'test'):
        split_data[split_name] = load_descriptor_split(
            base_path, split_name, encoder_descriptor_type, decoder_descriptor_type, n_elements
        )
    return split_data


def load_descriptor_data(base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, data_portion='train'):
    """
    Load descriptors for t-t mode.

    data_portion:
      - 'train': train.csv as the pool for runtime train/val split
      - 'test': test.csv as the held-out test set
    """
    if data_portion == 'train':
        split_names = ['train']
    elif data_portion == 'test':
        split_names = ['test']
    else:
        raise ValueError(f"Unsupported data_portion: {data_portion}. Expected 'train' or 'test'.")

    return _load_descriptor_from_splits(
        base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, split_names
    )


def prepare_descriptor_datasets(
    base_path,
    encoder_descriptor_type,
    decoder_descriptor_type,
    n_elements,
    seed=None,
):
    """Detect t-v-t / t-t from CSVs, load, preprocess."""
    if seed is None:
        seed = DATASET_CONFIG['seed']

    split_mode = detect_descriptor_split_mode(
        base_path, encoder_descriptor_type, decoder_descriptor_type
    )
    print(f"Detected descriptor split mode: {split_mode}")
    print(f"Structure placement: {get_structure_placement()}")
    if structure_dual_token_split():
        print(
            "Structure dual-token split: enabled "
            f"({structure_input_dim()}-d CSV -> {n_structure_tokens()} x {structure_per_token_dim()}-d tokens)"
        )
    if split_mode == 't-v-t':
        print("Loading precomputed train/val/test splits (t-v-t)...")
        split_data = load_all_descriptor_splits(
            base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements
        )
        decoder_feature_columns = split_data['train'][6]
        processed_data = preprocess_precomputed_splits(split_data, n_elements, seed=seed)
    else:
        print("Loading train pool and separate test split (t-t)...")
        train_pack = load_descriptor_data(
            base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, data_portion='train'
        )
        test_pack = load_descriptor_data(
            base_path, encoder_descriptor_type, decoder_descriptor_type, n_elements, data_portion='test'
        )
        processed_data = preprocess_data_once(
            train_pack[0],
            train_pack[1],
            train_pack[2],
            train_pack[3],
            train_pack[4],
            train_pack[5],
            train_pack[6],
            n_elements,
            seed=seed,
            test_encoder_data=test_pack[0],
            test_decoder_data=test_pack[1],
            test_structure_data=test_pack[2],
            test_targets=test_pack[3],
            test_formulas=test_pack[4],
            test_battery_ids=test_pack[5],
        )
        decoder_feature_columns = train_pack[6]

    return processed_data, decoder_feature_columns, split_mode

def calculate_dimensions(encoder_data, decoder_data, n_elements):
    features_per_element = DATASET_CONFIG['encoder_token_dim']
    expected_encoder_dim = n_elements * features_per_element
    if encoder_data.shape[1] != expected_encoder_dim:
        raise ValueError(f"Encoder feature dimension {encoder_data.shape[1]} does not match expected ({expected_encoder_dim} = {n_elements} * {features_per_element}). Please check data or DATASET_CONFIG['encoder_token_dim'].")

    decoder_token_dim = DATASET_CONFIG['decoder_token_dim']
    if decoder_data.shape[1] % decoder_token_dim != 0:
        raise ValueError(
            f"Property decoder feature dimension {decoder_data.shape[1]} is not divisible by "
            f"the feature dimension per token {decoder_token_dim}."
        )

    decoder_tokens = decoder_data.shape[1] // decoder_token_dim
    structure_dim = structure_per_token_dim() if uses_structure_input() else 0
    return features_per_element, decoder_tokens, decoder_token_dim, structure_dim


def _reshape_and_mask(encoder_data, decoder_data, formulas, n_elements, features_per_element, decoder_tokens, decoder_token_dim, structure_data=None):
    n_samples = encoder_data.shape[0]
    encoder_data = encoder_data.reshape(n_samples, n_elements, features_per_element)
    decoder_data = decoder_data.reshape(n_samples, decoder_tokens, decoder_token_dim)

    enc_seq_len = encoder_seq_len(n_elements)
    structure_offset = encoder_structure_offset()

    src_key_padding_mask = np.ones((n_samples, enc_seq_len), dtype=bool)
    if structure_offset:
        src_key_padding_mask[:, :structure_offset] = False

    for i, formula in enumerate(formulas):
        num_unique_elements = _count_unique_elements_in_formula(formula, n_elements)
        if num_unique_elements > n_elements:
            print(
                f"Warning: The number of elements in formula '{formula}' ({num_unique_elements}) "
                f"exceeds the maximum number of elements ({n_elements})."
            )
            num_unique_elements = n_elements
        src_key_padding_mask[i, structure_offset:structure_offset + num_unique_elements] = False

    return encoder_data, decoder_data, structure_data, src_key_padding_mask


def _make_materials_dataset(
    encoder_data,
    decoder_data,
    targets,
    src_key_padding_mask,
    formulas,
    battery_ids,
    structure_data=None,
):
    return MaterialsDataset(
        encoder_data,
        decoder_data,
        targets,
        src_key_padding_mask,
        formulas,
        battery_ids,
        structure_data,
    )


def _normalize_split_arrays(encoder_data, decoder_data, structure_data, targets):
    encoder_mean = np.mean(encoder_data, axis=0)
    encoder_std = np.std(encoder_data, axis=0)
    decoder_mean = np.mean(decoder_data, axis=0)
    decoder_std = np.std(decoder_data, axis=0)
    targets_mean = np.mean(targets)
    targets_std = np.std(targets)

    structure_mean = None
    structure_std = None
    if structure_data is not None:
        structure_mean, structure_std = _structure_normalization_stats(structure_data)

    _report_zero_variance_features(encoder_std, 'encoder')
    _report_zero_variance_features(decoder_std, 'decoder property')
    if structure_std is not None:
        if structure_data.ndim == 3:
            _report_zero_variance_features(structure_std.reshape(-1), 'structure')
        else:
            _report_zero_variance_features(structure_std, 'structure')

    encoder_data = _safe_standardize(encoder_data, encoder_mean, encoder_std)
    decoder_data = _safe_standardize(decoder_data, decoder_mean, decoder_std)
    targets_scaled = _safe_standardize(targets, targets_mean, targets_std)
    if structure_data is not None:
        structure_data = _standardize_structure_array(structure_data, structure_mean, structure_std)

    for name, arr in (
        ('encoder', encoder_data),
        ('decoder property', decoder_data),
        ('targets', targets_scaled),
        ('structure', structure_data),
    ):
        if arr is None:
            continue
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values detected in normalized {name} data. Check input descriptors.")

    stats = {
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'targets_mean': targets_mean,
        'targets_std': targets_std,
        'structure_mean': structure_mean,
        'structure_std': structure_std,
    }
    return encoder_data, decoder_data, structure_data, targets_scaled, stats


def preprocess_precomputed_splits(split_data, n_elements, seed=None):
    """Preprocess pre-split train/val/test CSVs for t-v-t mode."""
    if seed is None:
        seed = DATASET_CONFIG['seed']

    train_pack = split_data['train']

    decoder_feature_columns = train_pack[6]
    features_per_element, decoder_tokens, decoder_token_dim, structure_dim = calculate_dimensions(
        train_pack[0], train_pack[1], n_elements
    )

    split_arrays = {}
    for split_name, pack in split_data.items():
        encoder_data, decoder_data, structure_data, targets, formulas, battery_ids, _ = pack[:7]
        split_arrays[split_name] = _reshape_and_mask(
            encoder_data, decoder_data, formulas, n_elements,
            features_per_element, decoder_tokens, decoder_token_dim,
            structure_data=structure_data,
        )

    train_encoder, train_decoder, train_structure, train_mask = split_arrays['train']
    train_targets = split_data['train'][3]

    train_encoder, train_decoder, train_structure, train_targets_scaled, stats = _normalize_split_arrays(
        train_encoder, train_decoder, train_structure, train_targets
    )

    def transform_split(split_name, raw_targets):
        enc, dec, struct, mask = split_arrays[split_name]
        enc = _safe_standardize(enc, stats['encoder_mean'], stats['encoder_std'])
        dec = _safe_standardize(dec, stats['decoder_mean'], stats['decoder_std'])
        tgt = _safe_standardize(raw_targets, stats['targets_mean'], stats['targets_std'])
        if struct is not None:
            struct = _standardize_structure_array(struct, stats['structure_mean'], stats['structure_std'])
        return enc, dec, struct, mask, tgt

    val_encoder, val_decoder, val_structure, val_mask, val_targets_scaled = transform_split(
        'val', split_data['val'][3]
    )
    test_encoder, test_decoder, test_structure, test_mask, test_targets_scaled = transform_split(
        'test', split_data['test'][3]
    )

    count_stats = {}

    train_dataset = _make_materials_dataset(
        train_encoder, train_decoder, train_targets_scaled, train_mask,
        split_data['train'][4], split_data['train'][5], train_structure,
    )
    val_dataset = _make_materials_dataset(
        val_encoder, val_decoder, val_targets_scaled, val_mask,
        split_data['val'][4], split_data['val'][5], val_structure,
    )
    test_dataset = _make_materials_dataset(
        test_encoder, test_decoder, test_targets_scaled, test_mask,
        split_data['test'][4], split_data['test'][5], test_structure,
    )

    n_train = len(split_data['train'][4])
    n_val = len(split_data['val'][4])
    n_test = len(split_data['test'][4])
    all_formulas = np.concatenate([split_data['train'][4], split_data['val'][4], split_data['test'][4]])
    all_battery_ids = np.concatenate([split_data['train'][5], split_data['val'][5], split_data['test'][5]])

    print(f"Data split (precomputed): Train={n_train}, Val={n_val}, Test={n_test}")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_indices': np.arange(0, n_train),
        'val_indices': np.arange(n_train, n_train + n_val),
        'test_indices': np.arange(n_train + n_val, n_train + n_val + n_test),
        'formulas': all_formulas,
        'battery_ids': all_battery_ids,
        'features_per_element': features_per_element,
        'decoder_tokens': decoder_tokens,
        'decoder_token_dim': decoder_token_dim,
        'structure_dim': structure_dim,
        'structure_placement': get_structure_placement(),
        'encoder_seq_len': encoder_seq_len(n_elements),
        'test_dataset_full': None,
        **stats,
        **count_stats,
    }


def preprocess_data_once(
    encoder_data,
    decoder_data,
    structure_data,
    targets,
    formulas,
    battery_ids,
    decoder_feature_columns,
    n_elements,
    seed=None,
    test_encoder_data=None,
    test_decoder_data=None,
    test_structure_data=None,
    test_targets=None,
    test_formulas=None,
    test_battery_ids=None,
):
    """Preprocess t-t mode: split train.csv pool 0.875/0.125, test from test.csv."""
    if seed is None:
        seed = DATASET_CONFIG['seed']

    features_per_element, decoder_tokens, decoder_token_dim, structure_dim = calculate_dimensions(
        encoder_data, decoder_data, n_elements
    )

    encoder_data, decoder_data, structure_data, src_key_padding_mask = _reshape_and_mask(
        encoder_data, decoder_data, formulas, n_elements,
        features_per_element, decoder_tokens, decoder_token_dim,
        structure_data=structure_data,
    )
    n_samples = encoder_data.shape[0]

    if (
        test_encoder_data is None or test_decoder_data is None or test_targets is None
        or test_formulas is None or test_battery_ids is None
    ):
        raise ValueError("t-t mode requires a separate test set to be provided.")

    n_test_samples = test_encoder_data.shape[0]
    test_encoder_data, test_decoder_data, test_structure_data, test_src_key_padding_mask = _reshape_and_mask(
        test_encoder_data, test_decoder_data, test_formulas, n_elements,
        features_per_element, decoder_tokens, decoder_token_dim,
        structure_data=test_structure_data,
    )

    q_bins = 6
    while True:
        try:
            bins = pd.qcut(targets, q=q_bins, labels=False, duplicates='drop')
            break
        except ValueError:
            q_bins -= 1
            if q_bins < 2:
                raise ValueError("Unable to effectively bin average_voltage. Please check the data distribution or reduce the number of bins.")

    val_size = TT_POOL_VAL_RATIO / (TT_POOL_TRAIN_RATIO + TT_POOL_VAL_RATIO)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_indices, val_indices = next(sss.split(np.zeros(len(targets)), bins))
    test_indices = np.arange(n_test_samples)

    print(
        f"Data split (t-t, pool ratio {TT_POOL_TRAIN_RATIO}:{TT_POOL_VAL_RATIO}): "
        f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )

    encoder_train_data = encoder_data[train_indices]
    decoder_train_data = decoder_data[train_indices]
    targets_train_data = targets[train_indices]
    structure_train_data = structure_data[train_indices] if structure_data is not None else None

    encoder_mean = np.mean(encoder_train_data, axis=0)
    encoder_std = np.std(encoder_train_data, axis=0)
    decoder_mean = np.mean(decoder_train_data, axis=0)
    decoder_std = np.std(decoder_train_data, axis=0)
    targets_mean = np.mean(targets_train_data)
    targets_std = np.std(targets_train_data)

    structure_mean = None
    structure_std = None
    if structure_train_data is not None:
        structure_mean, structure_std = _structure_normalization_stats(structure_train_data)

    _report_zero_variance_features(encoder_std, 'encoder')
    _report_zero_variance_features(decoder_std, 'decoder property')
    if structure_std is not None:
        if structure_train_data is not None and structure_train_data.ndim == 3:
            _report_zero_variance_features(structure_std.reshape(-1), 'structure')
        else:
            _report_zero_variance_features(structure_std, 'structure')

    encoder_data = _safe_standardize(encoder_data, encoder_mean, encoder_std)
    decoder_data = _safe_standardize(decoder_data, decoder_mean, decoder_std)
    targets_scaled = _safe_standardize(targets, targets_mean, targets_std)

    if structure_data is not None:
        structure_data = _standardize_structure_array(structure_data, structure_mean, structure_std)

    for name, arr in (
        ('encoder', encoder_data),
        ('decoder property', decoder_data),
        ('targets', targets_scaled),
        ('structure', structure_data),
    ):
        if arr is None:
            continue
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values detected in normalized {name} data. Check input descriptors.")

    test_encoder_data = _safe_standardize(test_encoder_data, encoder_mean, encoder_std)
    test_decoder_data = _safe_standardize(test_decoder_data, decoder_mean, decoder_std)
    test_targets_scaled = _safe_standardize(test_targets, targets_mean, targets_std)
    if test_structure_data is not None:
        test_structure_data = _standardize_structure_array(
            test_structure_data, structure_mean, structure_std
        )

    test_dataset_full = _make_materials_dataset(
        test_encoder_data,
        test_decoder_data,
        test_targets_scaled,
        test_src_key_padding_mask,
        test_formulas,
        test_battery_ids,
        test_structure_data,
    )
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

    dataset = _make_materials_dataset(
        encoder_data,
        decoder_data,
        targets_scaled,
        src_key_padding_mask,
        formulas,
        battery_ids,
        structure_data,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    all_formulas = np.concatenate([formulas, test_formulas])
    all_battery_ids = np.concatenate([battery_ids, test_battery_ids])
    n_train_val = len(formulas)
    train_indices_global = train_indices
    val_indices_global = val_indices
    test_indices_global = np.arange(n_train_val, n_train_val + n_test_samples)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_indices': train_indices_global,
        'val_indices': val_indices_global,
        'test_indices': test_indices_global,
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'targets_mean': targets_mean,
        'targets_std': targets_std,
        'formulas': all_formulas,
        'battery_ids': all_battery_ids,
        'features_per_element': features_per_element,
        'decoder_tokens': decoder_tokens,
        'decoder_token_dim': decoder_token_dim,
        'structure_dim': structure_dim,
        'structure_mean': structure_mean,
        'structure_std': structure_std,
        'structure_placement': get_structure_placement(),
        'encoder_seq_len': encoder_seq_len(n_elements),
        'test_dataset_full': test_dataset_full,
    }


def create_data_loaders(processed_data, batch_size, seed=None):

    if seed is None:
        seed = DATASET_CONFIG['seed']
    
    train_loader = DataLoader(processed_data['train_dataset'], batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(processed_data['val_dataset'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(processed_data['test_dataset'], batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


def preprocess_data(encoder_data, decoder_data, structure_data, targets, formulas, battery_ids, decoder_feature_columns, n_elements, batch_size=32,
                    seed=None,
                    test_encoder_data=None, test_decoder_data=None, test_structure_data=None,
                    test_targets=None, test_formulas=None, test_battery_ids=None):
    """Preprocess t-t split data and return DataLoaders plus normalization stats."""
    processed_data = preprocess_data_once(
        encoder_data, decoder_data, structure_data, targets, formulas, battery_ids, decoder_feature_columns, n_elements,
        seed,
        test_encoder_data, test_decoder_data, test_structure_data, test_targets, test_formulas, test_battery_ids,
    )
    
    data_loaders = create_data_loaders(processed_data, batch_size, seed)

    return {
        'train_loader': data_loaders['train_loader'],
        'val_loader': data_loaders['val_loader'],
        'test_loader': data_loaders['test_loader'],
        'train_indices': processed_data['train_indices'],
        'val_indices': processed_data['val_indices'],
        'test_indices': processed_data['test_indices'],
        'encoder_mean': processed_data['encoder_mean'],
        'encoder_std': processed_data['encoder_std'],
        'decoder_mean': processed_data['decoder_mean'],
        'decoder_std': processed_data['decoder_std'],
        'targets_mean': processed_data['targets_mean'],
        'targets_std': processed_data['targets_std'],
        'formulas': processed_data['formulas'],
        'battery_ids': processed_data['battery_ids'],
        'features_per_element': processed_data['features_per_element'],
        'decoder_tokens': processed_data['decoder_tokens'],
        'decoder_token_dim': processed_data['decoder_token_dim'],
        'structure_dim': processed_data['structure_dim'],
        'structure_mean': processed_data['structure_mean'],
        'structure_std': processed_data['structure_std'],
        'structure_placement': processed_data.get('structure_placement', get_structure_placement()),
        'encoder_seq_len': processed_data.get('encoder_seq_len', encoder_seq_len(n_elements)),
    }
