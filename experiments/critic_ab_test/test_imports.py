#!/usr/bin/env python3
"""
Quick test of imports for A/B testing pipeline.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_imports():
    """Test all required imports."""
    print("🔄 Testing imports...")

    try:
        from experiments.critic_ab_test.critic_variants import (
            CRITIC_VARIANTS,
            TEST_ORDER,
        )

        print(f"  ✅ critic_variants loaded ({len(CRITIC_VARIANTS)} variants)")

        from src.meta_optimizer import (
            SectionalHyPEOptimizer,
            HypeMetaPromptConfig,
            PromptSectionSpec,
        )

        print("  ✅ SectionalHyPEOptimizer imported")

        # Check if _optimize_critic_with_config exists
        assert hasattr(SectionalHyPEOptimizer, "_optimize_critic_with_config"), (
            "Method _optimize_critic_with_config not found!"
        )
        print("  ✅ _optimize_critic_with_config method exists")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
