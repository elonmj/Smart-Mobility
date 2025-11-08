"""
Traffic Signal Controller with Safety Layer

Implements traffic signal phases, timing constraints, and safety logic
as specified in the design document.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Traffic signal states"""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    ALL_RED = "all_red"


@dataclass
class Phase:
    """Traffic signal phase definition"""
    id: int
    name: str
    description: str
    movements: List[str]
    

@dataclass
class Timings:
    """Signal timing constraints"""
    min_green: float = 10.0      # Minimum green time
    max_green: float = 120.0     # Maximum green time  
    yellow: float = 3.0          # Yellow duration
    all_red: float = 2.0         # All-red clearance
    

@dataclass
class SignalConfig:
    """Complete signal configuration"""
    phases: List[Phase]
    timings: Timings
    initial_phase: int = 0


class SignalController:
    """Traffic signal controller with safety layer"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.phases = {p.id: p for p in config.phases}
        self.timings = config.timings
        
        # Current state
        self.current_phase_id = config.initial_phase
        self.phase_start_time = 0.0
        self.current_signal_state = SignalState.GREEN
        self.state_start_time = 0.0
        
        # Safety tracking
        self.in_transition = False
        self.transition_target_phase = None
        
        # Logging
        self.phase_log = []
        
        logger.info(f"Initialized signal controller with {len(self.phases)} phases")
    
    def reset(self, timestamp: float = 0.0):
        """Reset controller to initial state"""
        self.current_phase_id = self.config.initial_phase
        self.phase_start_time = timestamp
        self.current_signal_state = SignalState.GREEN
        self.state_start_time = timestamp
        self.in_transition = False
        self.transition_target_phase = None
        self.phase_log = []
        
        logger.info(f"Reset signal controller at t={timestamp}")
    
    def get_current_phase(self) -> Phase:
        """Get current active phase"""
        return self.phases[self.current_phase_id]
    
    def get_phase_duration(self, timestamp: float) -> float:
        """Get how long current phase has been active"""
        return timestamp - self.phase_start_time
    
    def get_state_duration(self, timestamp: float) -> float:
        """Get how long current signal state has been active"""
        return timestamp - self.state_start_time
    
    def can_switch_phase(self, timestamp: float) -> bool:
        """Check if phase switch is allowed by safety constraints"""
        if self.in_transition:
            return False
            
        phase_duration = self.get_phase_duration(timestamp)
        
        # Must satisfy minimum green time
        if self.current_signal_state == SignalState.GREEN:
            return phase_duration >= self.timings.min_green
        
        return False
    
    def must_switch_phase(self, timestamp: float) -> bool:
        """Check if phase must be switched due to max green constraint"""
        if self.current_signal_state != SignalState.GREEN:
            return False
            
        phase_duration = self.get_phase_duration(timestamp)
        return phase_duration >= self.timings.max_green
    
    def request_phase_switch(self, timestamp: float) -> bool:
        """Request to switch to next phase (subject to safety checks)"""
        if not self.can_switch_phase(timestamp) and not self.must_switch_phase(timestamp):
            logger.debug(f"Phase switch denied at t={timestamp} (safety constraint)")
            return False
        
        # Start transition sequence
        next_phase_id = self._get_next_phase_id()
        self._start_transition(next_phase_id, timestamp)
        
        logger.info(f"Phase switch initiated: {self.current_phase_id} -> {next_phase_id} at t={timestamp}")
        return True
    
    def update(self, timestamp: float) -> Dict[str, Any]:
        """Update controller state and handle transitions"""
        if self.in_transition:
            self._handle_transition(timestamp)
        elif self.must_switch_phase(timestamp):
            # Force switch if max green time exceeded
            logger.warning(f"Forcing phase switch at t={timestamp} (max green exceeded)")
            self.request_phase_switch(timestamp)
            
        # Log phase changes
        phase_duration = self.get_phase_duration(timestamp)
        state_duration = self.get_state_duration(timestamp)
        
        return {
            "phase_id": self.current_phase_id,
            "phase_name": self.get_current_phase().name,
            "signal_state": self.current_signal_state.value,
            "phase_duration": phase_duration,
            "state_duration": state_duration,
            "in_transition": self.in_transition,
            "can_switch": self.can_switch_phase(timestamp),
            "must_switch": self.must_switch_phase(timestamp)
        }
    
    def _get_next_phase_id(self) -> int:
        """Get next phase ID in sequence"""
        phase_ids = sorted(self.phases.keys())
        current_index = phase_ids.index(self.current_phase_id)
        next_index = (current_index + 1) % len(phase_ids)
        return phase_ids[next_index]
    
    def _start_transition(self, target_phase_id: int, timestamp: float):
        """Start phase transition sequence"""
        self.in_transition = True
        self.transition_target_phase = target_phase_id
        
        # Start with yellow if currently green
        if self.current_signal_state == SignalState.GREEN:
            self._change_signal_state(SignalState.YELLOW, timestamp)
        else:
            # Already in transition, continue
            pass
    
    def _handle_transition(self, timestamp: float):
        """Handle ongoing phase transition"""
        state_duration = self.get_state_duration(timestamp)
        
        if self.current_signal_state == SignalState.YELLOW:
            if state_duration >= self.timings.yellow:
                self._change_signal_state(SignalState.ALL_RED, timestamp)
                
        elif self.current_signal_state == SignalState.ALL_RED:
            if state_duration >= self.timings.all_red:
                # Complete transition to new phase
                self._complete_transition(timestamp)
    
    def _complete_transition(self, timestamp: float):
        """Complete phase transition"""
        old_phase_id = self.current_phase_id
        self.current_phase_id = self.transition_target_phase
        self.phase_start_time = timestamp
        
        self._change_signal_state(SignalState.GREEN, timestamp)
        
        self.in_transition = False
        self.transition_target_phase = None
        
        # Log the phase change
        self.phase_log.append({
            "timestamp": timestamp,
            "from_phase": old_phase_id,
            "to_phase": self.current_phase_id,
            "duration": timestamp - (self.phase_start_time if hasattr(self, '_last_phase_start') else timestamp)
        })
        
        logger.info(f"Phase transition completed: {old_phase_id} -> {self.current_phase_id} at t={timestamp}")
    
    def _change_signal_state(self, new_state: SignalState, timestamp: float):
        """Change signal state"""
        old_state = self.current_signal_state
        self.current_signal_state = new_state
        self.state_start_time = timestamp
        
        logger.debug(f"Signal state changed: {old_state.value} -> {new_state.value} at t={timestamp}")
    
    def get_signal_plan(self) -> Dict[str, Any]:
        """Get current signal plan for ARZ endpoint"""
        return {
            "phase_id": self.current_phase_id,
            "phase_name": self.get_current_phase().name,
            "signal_state": self.current_signal_state.value,
            "movements": self.get_current_phase().movements,
            "in_transition": self.in_transition
        }
    
    def get_phase_timeline(self) -> List[Dict[str, Any]]:
        """Get phase change timeline for analysis"""
        return self.phase_log.copy()


def create_signal_controller(config_dict: Dict[str, Any]) -> SignalController:
    """Create signal controller from configuration dictionary"""
    phases = []
    for phase_data in config_dict["signals"]["phases"]:
        phases.append(Phase(
            id=phase_data["id"],
            name=phase_data["name"],
            description=phase_data["description"],
            movements=phase_data["movements"]
        ))
    
    timings = Timings(
        min_green=config_dict["signals"]["timings"]["min_green"],
        max_green=config_dict["signals"]["timings"]["max_green"],
        yellow=config_dict["signals"]["timings"]["yellow"],
        all_red=config_dict["signals"]["timings"]["all_red"]
    )
    
    config = SignalConfig(
        phases=phases,
        timings=timings,
        initial_phase=config_dict["signals"]["initial_phase"]
    )
    
    return SignalController(config)
