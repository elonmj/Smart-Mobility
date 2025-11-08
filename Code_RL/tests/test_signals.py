"""
Unit tests for the signal controller
"""

import pytest
import time
from src.signals.controller import (
    SignalController, Phase, Timings, SignalConfig, SignalState
)


class TestSignalController:
    def setup_method(self):
        # Create test configuration
        phases = [
            Phase(id=0, name="ns", description="North-South", movements=["n", "s"]),
            Phase(id=1, name="ew", description="East-West", movements=["e", "w"])
        ]
        timings = Timings(min_green=10.0, max_green=60.0, yellow=3.0, all_red=2.0)
        config = SignalConfig(phases=phases, timings=timings, initial_phase=0)
        
        self.controller = SignalController(config)
    
    def test_initialization(self):
        assert self.controller.current_phase_id == 0
        assert self.controller.current_signal_state == SignalState.GREEN
        assert not self.controller.in_transition
        assert len(self.controller.phases) == 2
    
    def test_get_current_phase(self):
        phase = self.controller.get_current_phase()
        assert phase.id == 0
        assert phase.name == "ns"
        assert phase.movements == ["n", "s"]
    
    def test_reset(self):
        # Modify state
        self.controller.current_phase_id = 1
        self.controller.in_transition = True
        
        # Reset
        self.controller.reset(timestamp=10.0)
        
        assert self.controller.current_phase_id == 0
        assert self.controller.phase_start_time == 10.0
        assert self.controller.current_signal_state == SignalState.GREEN
        assert not self.controller.in_transition
    
    def test_phase_duration_calculation(self):
        self.controller.reset(timestamp=0.0)
        duration = self.controller.get_phase_duration(15.0)
        assert duration == 15.0
    
    def test_cannot_switch_before_min_green(self):
        self.controller.reset(timestamp=0.0)
        # Try to switch after 5 seconds (< min_green of 10s)
        assert not self.controller.can_switch_phase(5.0)
    
    def test_can_switch_after_min_green(self):
        self.controller.reset(timestamp=0.0)
        # Can switch after min_green period
        assert self.controller.can_switch_phase(15.0)
    
    def test_must_switch_after_max_green(self):
        self.controller.reset(timestamp=0.0)
        # Must switch after max_green period
        assert self.controller.must_switch_phase(65.0)
    
    def test_successful_phase_switch_request(self):
        self.controller.reset(timestamp=0.0)
        
        # Request switch after min_green
        result = self.controller.request_phase_switch(15.0)
        assert result is True
        assert self.controller.in_transition is True
        assert self.controller.current_signal_state == SignalState.YELLOW
    
    def test_denied_phase_switch_request(self):
        self.controller.reset(timestamp=0.0)
        
        # Request switch before min_green
        result = self.controller.request_phase_switch(5.0)
        assert result is False
        assert self.controller.in_transition is False
        assert self.controller.current_signal_state == SignalState.GREEN
    
    def test_transition_sequence(self):
        self.controller.reset(timestamp=0.0)
        
        # Start transition
        self.controller.request_phase_switch(15.0)
        assert self.controller.current_signal_state == SignalState.YELLOW
        
        # Update during yellow phase
        self.controller.update(16.0)
        assert self.controller.current_signal_state == SignalState.YELLOW
        
        # Transition to all-red after yellow duration
        self.controller.update(18.1)  # 15 + 3.0 + 0.1
        assert self.controller.current_signal_state == SignalState.ALL_RED
        
        # Complete transition after all-red
        self.controller.update(20.1)  # + 2.0 + 0.1
        assert self.controller.current_signal_state == SignalState.GREEN
        assert self.controller.current_phase_id == 1  # Next phase
        assert not self.controller.in_transition
    
    def test_get_next_phase_id(self):
        # Test cycling through phases
        assert self.controller._get_next_phase_id() == 1  # From 0 to 1
        
        self.controller.current_phase_id = 1
        assert self.controller._get_next_phase_id() == 0  # From 1 back to 0
    
    def test_update_method(self):
        self.controller.reset(timestamp=0.0)
        
        info = self.controller.update(10.0)
        
        assert info["phase_id"] == 0
        assert info["phase_name"] == "ns"
        assert info["signal_state"] == "green"
        assert info["phase_duration"] == 10.0
        assert info["can_switch"] is True  # At min_green threshold
        assert info["must_switch"] is False
        assert info["in_transition"] is False
    
    def test_forced_switch_on_max_green(self):
        self.controller.reset(timestamp=0.0)
        
        # Update beyond max_green - should force switch
        info = self.controller.update(65.0)
        
        # Should have automatically started transition
        assert self.controller.in_transition is True
        assert info["must_switch"] is True
    
    def test_get_signal_plan(self):
        plan = self.controller.get_signal_plan()
        
        assert plan["phase_id"] == 0
        assert plan["phase_name"] == "ns"
        assert plan["signal_state"] == "green"
        assert plan["movements"] == ["n", "s"]
        assert plan["in_transition"] is False
    
    def test_phase_logging(self):
        self.controller.reset(timestamp=0.0)
        
        # Complete a phase transition
        self.controller.request_phase_switch(15.0)
        
        # Simulate full transition
        self.controller._change_signal_state(SignalState.YELLOW, 15.0)
        self.controller._change_signal_state(SignalState.ALL_RED, 18.0)
        self.controller._complete_transition(20.0)
        
        timeline = self.controller.get_phase_timeline()
        assert len(timeline) == 1
        assert timeline[0]["from_phase"] == 0
        assert timeline[0]["to_phase"] == 1
        assert timeline[0]["timestamp"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__])
