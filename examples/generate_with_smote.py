#!/usr/bin/env python
"""
Example: Generate synthetic data with optional SMOTE enhancement and auto-merging.

This demonstrates the enhanced generation capabilities:
1. Knowledge-based generation (Phase 4-5)
2. Optional SMOTE oversampling for more diversity
3. Automatic merging with original data
4. Auto-saving to data/generated/ folder
"""

import sys
from pathlib import Path

# Add project root to path
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
    """Generate synthetic banking data with SMOTE and merging."""
    
    logger.info("="*60)
    logger.info("Neural Sentinel - Enhanced Data Generation")
    logger.info("="*60)
    
    # Load knowledge base
    logger.info("\n1. Loading knowledge base...")
    knowledge = load_knowledge_base(project_root)
    logger.info("✓ Knowledge base loaded")
    
    # =========================================================================
    # OPTION 1: Basic generation (knowledge-based only, no SMOTE, no merge)
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("OPTION 1: Basic Generation (Knowledge-Based)")
    logger.info("="*60)
    
    account_gen = AccountGenerator(knowledge, seed=42)
    basic_accounts = account_gen.generate(
        n=1000,
        use_smote=False,
        merge_with_original=False
    )
    logger.info(f"✓ Generated {len(basic_accounts)} basic accounts")
    
    # =========================================================================
    # OPTION 2: Generation with SMOTE (adds 20% more via interpolation)
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("OPTION 2: Generation with SMOTE Enhancement")
    logger.info("="*60)
    
    account_gen_smote = AccountGenerator(knowledge, seed=42)
    smote_accounts = account_gen_smote.generate(
        n=1000,
        use_smote=True,  # Enable SMOTE
        merge_with_original=False,
        output_path=project_root / "data" / "generated" / "accounts_with_smote.csv"
    )
    logger.info(f"✓ Generated {len(smote_accounts)} accounts with SMOTE")
    
    # =========================================================================
    # OPTION 3: Generation + Merge with Original (default behavior)
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("OPTION 3: Generation + Merge with Original Data")
    logger.info("="*60)
    
    account_gen_merged = AccountGenerator(knowledge, seed=42)
    merged_accounts = account_gen_merged.generate(
        n=1000,
        use_smote=False,
        merge_with_original=True,  # Enable merging (default)
        # output_path auto-generated: data/generated/synthetic_accounts.csv
    )
    logger.info(f"✓ Generated and merged: {len(merged_accounts)} total accounts")
    if 'data_source' in merged_accounts.columns:
        logger.info(f"  - Original: {(merged_accounts['data_source'] == 'original').sum()}")
        logger.info(f"  - Synthetic: {(merged_accounts['data_source'] == 'synthetic').sum()}")
    
    # =========================================================================
    # OPTION 4: Full Enhancement (SMOTE + Merge)
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("OPTION 4: Full Enhancement (SMOTE + Merge)")
    logger.info("="*60)
    
    account_gen_full = AccountGenerator(knowledge, seed=42)
    full_accounts = account_gen_full.generate(
        n=1000,
        use_smote=True,               # Apply SMOTE
        merge_with_original=True,     # Merge with original
        output_path=project_root / "data" / "generated" / "accounts_full_enhanced.csv"
    )
    logger.info(f"✓ Fully enhanced: {len(full_accounts)} total accounts")
    
    # =========================================================================
    # Transaction Generation Examples
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Transaction Generation")
    logger.info("="*60)
    
    # Use basic accounts for transaction generation
    tx_gen = TransactionGenerator(knowledge, basic_accounts, seed=42)
    
    # Basic transactions
    logger.info("\n--- Basic Transactions ---")
    basic_tx = tx_gen.generate(
        n=5000,
        use_smote=False,
        merge_with_original=False
    )
    logger.info(f"✓ Generated {len(basic_tx)} basic transactions")
    
    # Transactions with SMOTE
    logger.info("\n--- Transactions with SMOTE ---")
    tx_gen_smote = TransactionGenerator(knowledge, basic_accounts, seed=42)
    smote_tx = tx_gen_smote.generate(
        n=5000,
        use_smote=True,
        merge_with_original=False,
        output_path=project_root / "data" / "generated" / "transactions_with_smote.csv"
    )
    logger.info(f"✓ Generated {len(smote_tx)} transactions with SMOTE")
    
    # Transactions merged with original
    logger.info("\n--- Transactions Merged with Original ---")
    tx_gen_merged = TransactionGenerator(knowledge, basic_accounts, seed=42)
    merged_tx = tx_gen_merged.generate(
        n=5000,
        use_smote=False,
        merge_with_original=True,
        # Auto-saves to data/generated/synthetic_transactions.csv
    )
    logger.info(f"✓ Generated and merged: {len(merged_tx)} total transactions")
    if 'data_source' in merged_tx.columns:
        logger.info(f"  - Original: {(merged_tx['data_source'] == 'original').sum()}")
        logger.info(f"  - Synthetic: {(merged_tx['data_source'] == 'synthetic').sum()}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Generation Complete - Files Created:")
    logger.info("="*60)
    
    generated_dir = project_root / "data" / "generated"
    if generated_dir.exists():
        for file in sorted(generated_dir.glob("*.csv")):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {file.name}: {size_mb:.2f} MB")
    
    logger.info("\n" + "="*60)
    logger.info("Usage Summary:")
    logger.info("="*60)
    logger.info("""
    # Knowledge-based only (fast, deterministic)
    accounts = gen.generate(n=1000, use_smote=False, merge_with_original=False)
    
    # With SMOTE enhancement (+20% via interpolation)
    accounts = gen.generate(n=1000, use_smote=True, merge_with_original=False)
    
    # Merged with original data (default behavior)
    accounts = gen.generate(n=1000, merge_with_original=True)
    
    # Full enhancement (SMOTE + merge + auto-save)
    accounts = gen.generate(n=1000, use_smote=True, merge_with_original=True)
    """)

if __name__ == "__main__":
    main()
