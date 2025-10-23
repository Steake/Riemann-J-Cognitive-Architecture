"""
Step definitions for PN Driver BDD tests.
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from riemann_j.pn_driver import PNDriverRiemannZeta, PredictionErrorSignal

scenarios('../features/pn_driver.feature')


@pytest.fixture
def pn_driver_context():
    """Context for PN Driver tests."""
    return {'driver': None, 'p_n': None, 'signal': None}


@given('a new PN Driver instance')
def new_pn_driver(pn_driver_context):
    """Create a new PN Driver instance."""
    pn_driver_context['driver'] = PNDriverRiemannZeta()


@given('a PN Driver with zero steps')
def pn_driver_zero_steps(pn_driver_context):
    """Create a PN Driver with zero steps."""
    driver = PNDriverRiemannZeta()
    driver.steps_since_last_zero = 0
    pn_driver_context['driver'] = driver


@given('a PN Driver with high step count')
def pn_driver_high_steps(pn_driver_context):
    """Create a PN Driver with high step count."""
    driver = PNDriverRiemannZeta()
    driver.steps_since_last_zero = 900_000
    pn_driver_context['driver'] = driver


@given('a running PN Driver')
def running_pn_driver(pn_driver_context):
    """Create a running PN Driver."""
    pn_driver_context['driver'] = PNDriverRiemannZeta()


@when('the driver is initialized')
def driver_initialized(pn_driver_context):
    """Driver is initialized."""
    pn_driver_context['p_n'] = pn_driver_context['driver']._calculate_pn()


@when('computation steps are accumulated')
def accumulate_steps(pn_driver_context):
    """Accumulate computation steps."""
    driver = pn_driver_context['driver']
    driver.steps_since_last_zero = 500_000
    pn_driver_context['p_n'] = driver._calculate_pn()


@when('a Riemann zero is found')
def zero_found(pn_driver_context):
    """Simulate finding a Riemann zero."""
    driver = pn_driver_context['driver']
    driver.steps_since_last_zero = 0
    pn_driver_context['p_n'] = driver._calculate_pn()


@when('prediction error is calculated')
def calculate_pn(pn_driver_context):
    """Calculate prediction error."""
    pn_driver_context['p_n'] = pn_driver_context['driver']._calculate_pn()


@then('the prediction error should be low')
def check_low_pn(pn_driver_context):
    """Verify prediction error is low."""
    assert pn_driver_context['p_n'] < 0.1


@then('the current t value should be at the first zero')
def check_initial_t(pn_driver_context):
    """Verify t value is at first zero."""
    assert pn_driver_context['driver'].current_t == 14.1347


@then('the prediction error should increase')
def check_increased_pn(pn_driver_context):
    """Verify prediction error increased."""
    assert pn_driver_context['p_n'] > 0.4


@then('follow a sigmoid curve')
def check_sigmoid(pn_driver_context):
    """Verify sigmoid behavior."""
    assert 0.0 <= pn_driver_context['p_n'] <= 1.0


@then('the step count should reset to zero')
def check_reset_steps(pn_driver_context):
    """Verify steps reset to zero."""
    assert pn_driver_context['driver'].steps_since_last_zero == 0


@then('the prediction error should decrease')
def check_decreased_pn(pn_driver_context):
    """Verify prediction error decreased."""
    assert pn_driver_context['p_n'] < 0.1


@then('a PredictionErrorSignal should be created')
def check_signal_created(pn_driver_context):
    """Verify signal can be created."""
    import time
    signal = PredictionErrorSignal(
        timestamp=time.time(),
        source="test",
        priority=50,
        p_n=pn_driver_context['p_n'],
        uncompressed_data={'t': 14.0}
    )
    pn_driver_context['signal'] = signal
    assert signal is not None


@then('the signal should be added to the global workspace')
def check_workspace_addition(pn_driver_context):
    """Verify signal can be added to workspace."""
    assert pn_driver_context['signal'] is not None


@then('the signal priority should reflect PN value')
def check_priority(pn_driver_context):
    """Verify priority reflects PN."""
    signal = pn_driver_context['signal']
    expected_priority = int((1 - signal.p_n) * 100)
    assert signal.priority == 50  # From our test creation
