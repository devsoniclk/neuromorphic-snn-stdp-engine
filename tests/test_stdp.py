import math

from neuromorphic_snn_stdp.stdp import EligibilityTrace, HomeostaticManager, STDPEngine


def test_pair_stdp_ltp():
    engine = STDPEngine()
    dw = engine.compute_pair_delta(10.0)
    assert dw > 0.0
    expected = 0.01 * math.exp(-10.0 / 20.0)
    assert math.isclose(dw, expected, rel_tol=1e-4)


def test_pair_stdp_ltd():
    engine = STDPEngine()
    dw = engine.compute_pair_delta(-10.0)
    assert dw < 0.0
    expected = -0.0105 * math.exp(-10.0 / 20.0)
    assert math.isclose(dw, expected, rel_tol=1e-4)


def test_pair_stdp_zero():
    engine = STDPEngine()
    assert engine.compute_pair_delta(0.0) == 0.0


def test_triplet_stdp_step():
    engine = STDPEngine()
    dw_post = engine.compute_triplet_step(r1=1.0, r2=0.5, o1=0.2, o2=0.8, is_post_spike=True)
    assert dw_post > 0.0
    dw_pre = engine.compute_triplet_step(r1=1.0, r2=0.5, o1=0.2, o2=0.8, is_post_spike=False)
    assert dw_pre < 0.0


def test_weight_clamping():
    engine = STDPEngine()
    assert engine.clamp(-0.5) == 0.0
    assert engine.clamp(1.5) == 1.0
    assert engine.clamp(0.42) == 0.42


def test_eligibility_trace():
    trace = EligibilityTrace(tau_eligibility_ms=50.0)
    trace.add_event(0.2)
    assert trace.trace == 0.2
    trace.decay(50.0)
    assert math.isclose(trace.trace, 0.2 * math.exp(-1.0), rel_tol=1e-4)
    update = trace.compute_reward_update(reward=1.0, learning_rate=0.5)
    assert update > 0.0


def test_homeostatic_scaling():
    manager = HomeostaticManager()
    dw_low = manager.multiplicative_scale(weight=0.5, current_rate_hz=5.0)
    assert dw_low > 0.0
    dw_high = manager.multiplicative_scale(weight=0.5, current_rate_hz=25.0)
    assert dw_high < 0.0
    thresh = manager.bcm_threshold([10.0, 20.0])
    assert thresh > 0.0
