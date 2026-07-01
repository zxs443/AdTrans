"""Export finetuned CHGNet structure descriptors."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

DATA_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_ROOT))

from chgnet_finetune_dual.chgnet_descriptor import generate_dual_structure_descriptors
from chgnet_finetune_dual.config import DualStateFinetuneConfig
from chgnet_finetune_dual.element_split import DescriptorSettings, apply_descriptor_settings
from chgnet_finetune_dual.split import load_split_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_dual_finetuned_descriptors(config: DualStateFinetuneConfig) -> None:
    config = config.resolve_paths(DATA_PROCESSING_ROOT)
    apply_descriptor_settings(
        DescriptorSettings(
            max_elements=config.max_elements,
            descriptor_root=Path(config.descriptor_root),
        )
    )

    split_assignment = load_split_assignment(config.electrode_data_path)

    checkpoint = config.resolved_finetune_checkpoint()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Fine-tuned backbone not found: {checkpoint}")

    generate_dual_structure_descriptors(
        config.electrode_data_path,
        split_assignment,
        chgnet_checkpoint=str(checkpoint),
        chgnet_model_name=config.chgnet_model_name,
        chgnet_device=config.export_device or None,
        export_embedding_mode=config.export_embedding_mode,
        structure_descriptor_name=config.export_descriptor_name,
        structure_column_prefix=config.structure_column_prefix,
        descriptor_root=Path(config.descriptor_root),
        graph_cache_dir=config.graph_cache_dir,
    )


def main() -> None:
    export_dual_finetuned_descriptors(DualStateFinetuneConfig())


if __name__ == "__main__":
    main()
