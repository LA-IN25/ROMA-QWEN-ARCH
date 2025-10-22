"""
Configuration Loader for Agents and Profiles

This module handles loading, validating, and caching of agent and profile configurations
from YAML files. It uses Pydantic models for robust validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import os  # Import the 'os' module

from .models import AgentsYAMLConfig, ProfileYAMLConfig, validate_agents_yaml, validate_profile_yaml

# A simple in-memory cache for loaded configurations
_config_cache: Dict[Path, Any] = {}


class AgentConfigLoader:
    """Loads, validates, and provides access to agent configurations."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the loader with the path to the main agents.yaml file.
        
        Args:
            config_path: Path to the agents.yaml file
        """
        if not config_path.is_file():
            raise FileNotFoundError(f"Agent configuration file not found: {config_path}")
        self.config_path = config_path
        
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load YAML file content with environment variable substitution.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            Loaded YAML content as a dictionary
        """
        if path in _config_cache:
            logger.debug(f"Loading cached config from {path}")
            return _config_cache[path]
            
        logger.info(f"Loading agent configuration from: {path}")
        
        try:
            # Read the raw YAML file content as a string
            raw_content = path.read_text()
            
            # Get the model ID from environment variable, with a fallback
            local_model_id = os.getenv("LOCAL_MODEL_ID", "openai/default-local-model")
            
            # Substitute the placeholder with the environment variable
            # This allows easy model switching from the .env file
            substituted_content = raw_content.replace("${LOCAL_MODEL_ID}", local_model_id)
            
            # Parse the substituted string as YAML
            config_dict = yaml.safe_load(substituted_content)
            
            _config_cache[path] = config_dict
            return config_dict
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load or process YAML file {path}: {e}")
            raise

    def load_config(self) -> AgentsYAMLConfig:
        """
        Load and validate the main agents.yaml configuration.
        
        Returns:
            Validated AgentsYAMLConfig object
        """
        try:
            config_dict = self._load_yaml(self.config_path)
            validated_config = validate_agents_yaml(config_dict)
            
            total_agents = len(validated_config.agents)
            logger.success(f"✅ Loaded and validated configuration for {total_agents} agents")
            
            return validated_config
            
        except Exception as e:
            logger.error(f"Failed to load agent configuration: {e}")
            raise

    def resolve_prompt(self, prompt_source: str) -> Optional[str]:
        """
        Dynamically resolve a prompt source string to its content.
        
        Args:
            prompt_source: A string like 'prompts.planner_prompts.SYSTEM_MESSAGE'
            
        Returns:
            The prompt content as a string, or None if not found
        """
        if not prompt_source:
            return None
            
        try:
            # Construct the full module path
            full_module_path = f"sentientresearchagent.hierarchical_agent_framework.agent_configs.{prompt_source}"
            
            parts = full_module_path.split('.')
            module_name = ".".join(parts[:-1])
            var_name = parts[-1]
            
            # Import the module and get the variable
            module = __import__(module_name, fromlist=[var_name])
            prompt_content = getattr(module, var_name, None)
            
            if prompt_content is None:
                logger.warning(f"Could not find prompt variable '{var_name}' in module '{module_name}'")
            
            return prompt_content
            
        except ImportError:
            logger.error(f"Could not import prompt module for source: {prompt_source}")
            return None

