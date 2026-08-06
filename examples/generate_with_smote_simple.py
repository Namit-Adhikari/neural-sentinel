#!/usr/bin/env python
"""
Simple example: Generate synthetic banking data using SMOTE.

SMOTE is the primary generator - it learns from original data and creates
new synthetic samples through k-nearest neighbor interpolation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.generation.core.knowledge_extractor import load_knowledge_base
from src.generation.core.account_generator import AccountGenerator
from src.generation.core.transaction_generator import TransactionGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Generate synthetic banking data using SMOTE."""
    
    logger.info("="*60)
    logger.info("Neural Sentinel - SMOTE Data Generation")
    logger.info("="*60)
    
    # Load knowledge base (for compatibility, not used by SMOTE)
    logger.info("\n1. Loading knowledge base...")
    knowledge = load_knowledge_base(project_root)
    
    # =========================================================================
    # Generate Accounts using SMOTE
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Generating Accounts with SMOTE")
    logger.info("="*60)
    
    account_gen = AccountGenerator(knowledge, seed=42)
    
    # Generate 10K synthetic accounts
    # By default:
    # - Loads original accounts from data/original/accounts.csv
    # - Fits SMOTE on original data
    # - Generates 10K new samples via k-NN interpolation
    # - Merges with original data
    # - Saves to data/generated/synthetic_accounts.csv
    
    accounts = account_gen.generate(n=10_000)
    
    logger.info(f"\n✓ Generated {len(accounts)} total accounts")
    if 'data_source' in accounts.columns:
        logger.info(f"  - Original: {(accounts['data_source'] == 'original').sum()}")
        logger.info(f"  - Synthetic (SMOTE): {(accounts['data_source'] == 'synthetic').sum()}")
    
    # =========================================================================
    # Generate Transactions using SMOTE
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Generating Transactions with SMOTE")
    logger.info("="*60)
    
    tx_gen = TransactionGenerator(knowledge, accounts, seed=42)
    
    # Generate 50K synthetic transactions
    # By default:
    # - Loads original transactions from data/interim/transactions.parquet
    # - Fits SMOTE on original data
    # - Generates 50K new samples
    # - Merges with original data
    # - Saves to data/generated/synthetic_transactions.csv
    
    transactions = tx_gen.generate(n=50_000)
    
    logger.info(f"\n✓ Generated {len(transactions)} total transactions")
    if 'data_source' in transactions.columns:
        logger.info(f"  - Original: {(transactions['data_source'] == 'original').sum()}")
        logger.info(f"  - Synthetic (SMOTE): {(transactions['data_source'] == 'synthetic').sum()}")
    
    # =========================================================================
    # Custom Options
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Custom Generation Options")
    logger.info("="*60)
    
    # Generate without merging (synthetic only)
    synthetic_only_accounts = account_gen.generate(
        n=5_000,
        merge_with_original=False,  # Don't merge with original
        output_path=project_root / "data" / "generated" / "synthetic_only_accounts.csv"
    )
    logger.info(f"\n✓ Generated {len(synthetic_only_accounts)} synthetic-only accounts")
    
    # Generate with custom original data path
    custom_tx = tx_gen.generate(
        n=10_000,
        merge_with_original=True,
        original_data_path=project_root / "data" / "interim" / "transactions.parquet",
        output_path=project_root / "data" / "generated" / "custom_transactions.csv"
    )
    logger.info(f"✓ Generated {len(custom_tx)} transactions with custom paths")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Generation Complete!")
    logger.info("="*60)
    
    logger.info("\nFiles created in data/generated/:")
    generated_dir = project_root / "data" / "generated"
    if generated_dir.exists():
        for file in sorted(generated_dir.glob("*.csv")):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {file.name}: {size_mb:.2f} MB")
    
    logger.info("\nSMOTE Benefits:")
    logger.info("  ✓ Best benchmarking performance")
    logger.info("  ✓ Fast generation (no training)")
    logger.info("  ✓ Preserves local data patterns")
    logger.info("  ✓ Handles mixed data types")
    logger.info("  ✓ Automatic merge with original data")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        logger.error("\nMake sure original data files exist:")
        logger.error("  - data/original/accounts.csv")
        logger.error("  - data/interim/transactions.parquet")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"\n❌ Import Error: {e}")
        logger.error("\nInstall required package:")
        logger.error("  pip install imbalanced-learn==0.12.0")
        sys.exit(1)
