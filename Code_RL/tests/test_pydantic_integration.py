"""
Quick test to validate Pydantic config integration in Code_RL.

This test ensures:
1. RLConfigBuilder can create configs
2. Pydantic ARZ configs are accessible
3. Environment can be created (mock mode)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add Code_RL/src to path for imports
code_rl_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(code_rl_src))

def test_rl_config_builder():
    """Test RLConfigBuilder creation"""
    from utils.config import RLConfigBuilder
    
    print("\n" + "="*60)
    print("TEST 1: RLConfigBuilder Creation")
    print("="*60)
    
    # Test simple scenario
    print("\n📝 Creating simple scenario config...")
    rl_config = RLConfigBuilder.for_training(
        scenario="simple",
        N=50,
        episode_length=100.0
    )
    
    print(f"✅ RLConfigBuilder created")
    print(f"   ARZ config type: {type(rl_config.arz_simulation_config).__name__}")
    print(f"   Grid size: N={rl_config.arz_simulation_config.grid.N}")
    print(f"   CFL number: {rl_config.arz_simulation_config.cfl_number}")
    print(f"   Device: {rl_config.arz_simulation_config.device}")
    
    # Test Lagos scenario
    print("\n📝 Creating Lagos scenario config...")
    rl_config_lagos = RLConfigBuilder.for_training(
        scenario="lagos",
        N=200,
        episode_length=3600.0,
        dt_decision=15.0
    )
    
    print(f"✅ Lagos config created")
    print(f"   Grid size: N={rl_config_lagos.arz_simulation_config.grid.N}")
    print(f"   Episode length: {rl_config_lagos.rl_env_params['episode_length']}s")
    print(f"   Decision interval: {rl_config_lagos.rl_env_params['dt_decision']}s")
    print(f"   Max density (motorcycles): {rl_config_lagos.rl_env_params['normalization']['rho_max_motorcycles']} veh/km")
    
    return True


def test_environment_creation():
    """Test environment creation with Pydantic configs"""
    from utils.config import RLConfigBuilder
    from rl.train_dqn import create_environment_pydantic
    
    print("\n" + "="*60)
    print("TEST 2: Environment Creation (Mock Mode)")
    print("="*60)
    
    print("\n📝 Creating config...")
    rl_config = RLConfigBuilder.for_training(
        scenario="simple",
        N=50,
        episode_length=100.0
    )
    
    print("\n📝 Creating environment...")
    env = create_environment_pydantic(rl_config, use_mock=True)
    
    print(f"✅ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Max steps: {env.config.max_steps}")
    
    # Test reset
    print("\n📝 Testing environment reset...")
    obs, info = env.reset()
    
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test step
    print("\n📝 Testing environment step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Environment step successful")
    print(f"   Action: {action}")
    print(f"   Reward: {reward:.3f}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}")
    
    return True


def test_pydantic_arz_config():
    """Test that Pydantic ARZ configs are valid"""
    from utils.config import RLConfigBuilder
    
    print("\n" + "="*60)
    print("TEST 3: Pydantic ARZ Config Validation")
    print("="*60)
    
    print("\n📝 Creating config...")
    rl_config = RLConfigBuilder.for_training(scenario="simple", N=100)
    
    arz_config = rl_config.arz_simulation_config
    
    print(f"✅ ARZ config validated by Pydantic")
    print(f"\n📊 Configuration Details:")
    print(f"   Grid: N={arz_config.grid.N}, domain=[{arz_config.grid.xmin}, {arz_config.grid.xmax}]")
    print(f"   IC type: {arz_config.initial_conditions.type}")
    print(f"   BC left: {arz_config.boundary_conditions.left.type}")
    print(f"   BC right: {arz_config.boundary_conditions.right.type}")
    print(f"   Time: t_final={arz_config.t_final}s, CFL={arz_config.cfl_number}")
    print(f"   Device: {arz_config.device}")
    
    return True


def test_legacy_compatibility():
    """Test that legacy dict conversion works"""
    from utils.config import RLConfigBuilder
    
    print("\n" + "="*60)
    print("TEST 4: Legacy YAML Compatibility")
    print("="*60)
    
    print("\n📝 Creating Pydantic config...")
    rl_config = RLConfigBuilder.for_training(scenario="lagos", N=200)
    
    print("\n📝 Converting to legacy dict format...")
    legacy_dict = rl_config.to_legacy_dict()
    
    print(f"✅ Conversion successful")
    print(f"   Keys: {list(legacy_dict.keys())}")
    print(f"   Has 'env': {'env' in legacy_dict}")
    print(f"   Has 'endpoint': {'endpoint' in legacy_dict}")
    print(f"   Has 'signals': {'signals' in legacy_dict}")
    print(f"   Has 'arz_simulation': {'arz_simulation' in legacy_dict}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  Code_RL Pydantic Integration Tests")
    print("="*70)
    
    tests = [
        ("RLConfigBuilder Creation", test_rl_config_builder),
        ("Environment Creation", test_environment_creation),
        ("Pydantic ARZ Config", test_pydantic_arz_config),
        ("Legacy Compatibility", test_legacy_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            import traceback
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Pydantic integration successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
