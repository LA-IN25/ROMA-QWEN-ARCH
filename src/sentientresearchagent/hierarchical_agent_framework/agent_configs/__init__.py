"""
Initialization for agent configuration modules.

This __init__ file makes key classes and functions available
for easier import from other parts of the agent framework.
"""

# Import key components for easier access
# Removed the non-existent 'load_agent_configs' import
from .config_loader import AgentConfigLoader
from .agent_factory import AgentFactory, create_agents_from_config
from .registry_integration import integrate_yaml_agents, YAMLIntegrationManager
from .profile_loader import ProfileLoader, load_profile_config
from .models import AgentConfig, ModelConfig, ProfileConfig, AgentsYAMLConfig, ProfileYAMLConfig

# Define what gets imported when using 'from .agent_configs import *'
__all__ = [
    "AgentConfigLoader",
    "AgentFactory",
    "create_agents_from_config",
    "integrate_yaml_agents",
    "YAMLIntegrationManager",
    "ProfileLoader",
    "load_profile_config",
    "AgentConfig",
    "ModelConfig",
    "ProfileConfig",
    "AgentsYAMLConfig",
    "ProfileYAMLConfig",
]

