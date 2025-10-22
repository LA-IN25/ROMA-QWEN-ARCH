"""
Agent Factory

Creates agent instances from YAML configuration with comprehensive Pydantic validation.
Leverages structured Pydantic models for type safety and validation.
Includes logic to enable JSON mode for models when a response_model is specified.
"""

import importlib
import os
from typing import Dict, Any, Optional, List, Union, Type
from loguru import logger
from pathlib import Path

try:
    from omegaconf import DictConfig
except ImportError:
    logger.error("OmegaConf not installed. Please install with: pip install omegaconf>=2.3.0")
    raise

try:
    from agno.agent import Agent as AgnoAgent
    from agno.models.litellm import LiteLLM
    from agno.models.openai import OpenAIChat
    # Import agno.tools module for dynamic tool discovery
    import agno.tools
except ImportError as e:
    logger.error(f"Agno dependencies not available: {e}")
    raise

# Try to import WikipediaTools, but don't fail if wikipedia package is missing
try:
    from agno.tools.wikipedia import WikipediaTools
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    logger.warning("WikipediaTools not available (missing wikipedia package)")
    WikipediaTools = None
    WIKIPEDIA_AVAILABLE = False

from ..agents.adapters import (
    PlannerAdapter, ExecutorAdapter, AtomizerAdapter,
    AggregatorAdapter, PlanModifierAdapter
)
from ..agents.definitions.custom_searchers import OpenAICustomSearchAdapter, GeminiCustomSearchAdapter
from ..agents.definitions.exa_searcher import ExaCustomSearchAdapter
from sentientresearchagent.hierarchical_agent_framework.context.agent_io_models import (
    PlanOutput, AtomizerOutput, WebSearchResultsOutput,
    CustomSearcherOutput, PlanModifierInput
)
from ..types import TaskType
from sentientresearchagent.hierarchical_agent_framework.agents.base_adapter import BaseAdapter
from sentientresearchagent.hierarchical_agent_framework.agents.registry import AgentRegistry
from sentientresearchagent.hierarchical_agent_framework.agent_blueprints import AgentBlueprint
from sentientresearchagent.hierarchical_agent_framework.toolkits.data import BinanceToolkit, CoinGeckoToolkit, ArkhamToolkit, DefiLlamaToolkit
from .models import (
    AgentConfig, ModelConfig, ToolConfig, ToolkitConfig,
    validate_agent_config, validate_toolkit_config
)

try:
    from dotenv import load_dotenv
    # Load .env file at module level to ensure environment variables are available
    load_dotenv()
    logger.debug("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed. Environment variables from .env files will not be loaded automatically.")

# --- Helper function to convert OmegaConf DictConfig to standard dict recursively ---
def _to_dict(obj):
    if isinstance(obj, DictConfig):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    else:
        return obj
# --- End Helper ---

class AgentFactory:
    """Factory for creating agent instances from configuration with structured output support."""

    def __init__(self, config_loader):
        """
        Initialize the agent factory.

        Args:
            config_loader: AgentConfigLoader instance for resolving prompts
        """
        self.config_loader = config_loader
        self._adapter_classes = {
            "PlannerAdapter": PlannerAdapter,
            "ExecutorAdapter": ExecutorAdapter,
            "AtomizerAdapter": AtomizerAdapter,
            "AggregatorAdapter": AggregatorAdapter,
            "PlanModifierAdapter": PlanModifierAdapter,
            "OpenAICustomSearchAdapter": OpenAICustomSearchAdapter,
            "GeminiCustomSearchAdapter": GeminiCustomSearchAdapter,
            "ExaCustomSearchAdapter": ExaCustomSearchAdapter,
        }

        # Enhanced response models mapping
        self._response_models = {
            "PlanOutput": PlanOutput,
            "AtomizerOutput": AtomizerOutput,
            "WebSearchResultsOutput": WebSearchResultsOutput,
            "CustomSearcherOutput": CustomSearcherOutput,
            # Add more as needed
        }

        # Custom toolkits (our own implementations)
        self._toolkits = {
            "BinanceToolkit": BinanceToolkit,
            "CoingeckoToolkit": CoinGeckoToolkit,
            "ArkhamToolkit": ArkhamToolkit,
            "DefiLlamaToolkit": DefiLlamaToolkit,
        }

        # Initialize tools mapping for individual tools (not toolkits)
        self._tools = {}
        self._initialize_common_tools()

        # Model providers mapping
        self._model_providers = {
            "litellm": LiteLLM,
            "openai": OpenAIChat,
        }

        # Log available toolkits for visibility
        custom_toolkits = list(self._toolkits.keys())
        logger.info(f"📦 Available custom toolkits: {custom_toolkits}")

    def _initialize_common_tools(self) -> None:
        """Initialize common tools mapping for individual tool access."""
        try:
            # Import common tools from agno.tools
            from agno.tools.python import PythonTools
            from agno.tools.e2b import E2BTools
            from agno.tools.reasoning import ReasoningTools

            self._tools.update({
                "PythonTools": PythonTools,
                "E2BTools": E2BTools,
                "ReasoningTools": ReasoningTools,
            })

            # Optional tools with graceful degradation
            try:
                from agno.tools.duckduckgo import DuckDuckGoTools
                self._tools["DuckDuckGoTools"] = DuckDuckGoTools
            except ImportError as e:
                logger.warning(f"DuckDuckGoTools not available: {e}")

            # Add WikipediaTools if available
            if WIKIPEDIA_AVAILABLE and WikipediaTools:
                self._tools["WikipediaTools"] = WikipediaTools

            logger.info(f"Initialized {len(self._tools)} common tools: {list(self._tools.keys())}")

        except ImportError as e:
            logger.warning(f"Could not initialize some common tools: {e}")

    def resolve_response_model(self, response_model_name: str) -> Optional[Type]:
        """
        Resolve a response model name to the actual Pydantic model class.

        Args:
            response_model_name: Name of the response model

        Returns:
            Pydantic model class or None
        """
        if not response_model_name:
            return None

        if response_model_name in self._response_models:
            return self._response_models[response_model_name]

        # Try to import dynamically if not in our mapping
        try:
            # Try importing from agent_io_models
            module = importlib.import_module(
                "sentientresearchagent.hierarchical_agent_framework.context.agent_io_models"
            )
            if hasattr(module, response_model_name):
                model_class = getattr(module, response_model_name)
                self._response_models[response_model_name] = model_class
                return model_class
        except ImportError:
            pass

        logger.warning(f"Unknown response model: {response_model_name}")
        return None

    def create_model_instance(self, model_config: Union[DictConfig, Dict[str, Any], ModelConfig]) -> Union[LiteLLM, OpenAIChat]:
        """
        Create a model instance from configuration with Pydantic validation.

        Args:
            model_config: Model configuration (DictConfig, dict, or ModelConfig)

        Returns:
            Model instance

        Raises:
            ValueError: If configuration is invalid or required API keys are missing
        """
        # Convert to Pydantic ModelConfig for validation
        if isinstance(model_config, ModelConfig):
            validated_config = model_config
        else:
            # Convert DictConfig or dict to dict using our helper
            config_dict = _to_dict(model_config)

            # Validate using Pydantic model (includes environment validation)
            try:
                validated_config = ModelConfig(**config_dict)
                logger.debug(f"✅ Model configuration validated: {validated_config.provider}/{validated_config.model_id}")
            except Exception as e:
                logger.error(f"Model configuration validation failed: {e}")
                raise ValueError(f"Invalid model configuration: {e}") from e

        provider = validated_config.provider
        model_id = validated_config.model_id

        if provider not in self._model_providers:
            raise ValueError(f"Unsupported model provider: {provider}. Available: {list(self._model_providers.keys())}")

        model_class = self._model_providers[provider]

        # Extract model parameters that should be passed to the model constructor
        model_kwargs = {"id": model_id}

        # Standard LLM parameters that models support
        supported_llm_params = [
            'temperature', 'max_tokens', 'top_p', 'top_k',
            'frequency_penalty', 'presence_penalty', 'repetition_penalty',
            'min_p', 'tfs', 'typical_p', 'epsilon_cutoff', 'eta_cutoff',
            'response_format' # <-- Add response_format here
        ]

        # Use validated config instead of raw model_config
        for param in supported_llm_params:
            param_value = getattr(validated_config, param, None)
            # Handle special case for response_format which might be a dict
            if param == 'response_format' and isinstance(param_value, dict):
                 model_kwargs[param] = param_value
                 logger.debug(f"Adding JSON mode parameter response_format={param_value} to {provider}/{model_id}")
            elif param_value is not None:
                model_kwargs[param] = param_value
                logger.debug(f"Adding model parameter {param}={param_value} to {provider}/{model_id}")

        try:
            if provider == "litellm":
                is_o3_model = "o3" in model_id.lower()

                if is_o3_model:
                    logger.info(f"🔧 Creating LiteLLM model for o3: {model_id} with global drop_params=True")
                    import litellm
                    litellm.drop_params = True

                logger.info(f"🔧 Creating LiteLLM model: {model_id}")
                return model_class(**model_kwargs)

            elif provider == "openai":
                logger.info(f"🔧 Creating OpenAI model: {model_id}")
                return model_class(**model_kwargs)

            elif provider in ["fireworks", "fireworks_ai"]:
                logger.info(f"🔧 Creating Fireworks AI model: {model_id}")
                return model_class(**model_kwargs)

            else:
                logger.info(f"🔧 Creating {provider} model: {model_id}")
                return model_class(**model_kwargs)

        except Exception as e:
            logger.error(f"Failed to create model instance for {provider}/{model_id}: {e}")
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
                raise ValueError(
                    f"Authentication failed for {provider}/{model_id}. "
                    f"Please verify your API key in .env file is correct and has the necessary permissions. "
                    f"Original error: {e}"
                ) from e
            elif "rate limit" in error_msg:
                raise ValueError(
                    f"Rate limit exceeded for {provider}/{model_id}. "
                    f"Please wait before retrying or check your API quota. "
                    f"Original error: {e}"
                ) from e
            elif "not found" in error_msg or "model" in error_msg:
                raise ValueError(
                    f"Model '{model_id}' not found or not accessible for provider '{provider}'. "
                    f"Please verify the model name is correct. "
                    f"Original error: {e}"
                ) from e
            else:
                raise ValueError(
                    f"Failed to create model instance for {provider}/{model_id}. "
                    f"Original error: {e}"
                ) from e

    def create_tools(self, tool_configs: List[Union[str, Dict[str, Any]]]) -> List[Any]:
        """
        Create tool instances from tool configurations.

        Args:
            tool_configs: List of tool names (strings) or tool configurations (dicts)
                         Dict format: {"name": "ToolName", "params": {...}}

        Returns:
            List of tool instances
        """
        tools = []
        web_search = None
        clean_tools_func = None

        # Check if web_search is needed and import if necessary
        tool_names = []
        for config in tool_configs:
            if isinstance(config, str):
                tool_names.append(config)
            elif isinstance(config, dict) or hasattr(config, '__getitem__'):
                 tool_names.append(config.get("name") or "")

        if "web_search" in tool_names:
             try:
                from ..tools.web_search_tool import web_search, clean_tools
                clean_tools_func = clean_tools
                logger.debug("Imported web_search function")
             except ImportError as e:
                logger.error(f"Failed to import web_search: {e}")


        for config in tool_configs:
            tool_name = ""
            tool_params = {}
            if isinstance(config, str):
                tool_name = config
            elif isinstance(config, dict) or hasattr(config, '__getitem__'):
                tool_name = config.get("name") or ""
                tool_params = _to_dict(config.get("params", {})) # Convert OmegaConf params
            else:
                logger.warning(f"Invalid tool configuration type: {type(config)}")
                continue

            if tool_name in self._tools:
                try:
                    tool_class = self._tools[tool_name]
                    if tool_name == "PythonTools" and "save_and_run" not in tool_params:
                        tool_params["save_and_run"] = False
                        logger.debug(f"Setting save_and_run=False for PythonTools (default)")

                    if tool_params:
                        tool_instance = tool_class(**tool_params)
                        logger.debug(f"Created tool: {tool_name} with params: {tool_params}")
                    else:
                        tool_instance = tool_class()
                        logger.debug(f"Created tool: {tool_name}")
                    tools.append(tool_instance)
                except Exception as e:
                    logger.error(f"Failed to create tool {tool_name} with params: {tool_params} - {e}")
            elif tool_name == "web_search" and web_search is not None:
                tools.append(web_search)
                logger.debug("Added web_search function as tool")
            else:
                logger.warning(f"Unknown tool: {tool_name}")

        if clean_tools_func:
            tools = clean_tools_func(tools)
            logger.debug("Cleaned tools to remove problematic attributes")

        return tools

    def create_toolkits(self, toolkit_configs: List[Dict[str, Any]]) -> List[Any]:
        """
        Create toolkit instances and extract specified tools.
        Supports both custom toolkits and agno toolkits with dynamic discovery.

        Args:
            toolkit_configs: List of toolkit configurations

        Returns:
            List of selected toolkits

        Raises:
            ValueError: If toolkit configuration is invalid
        """
        selected_toolkits = []

        for config in toolkit_configs:
             # Convert OmegaConf DictConfig to standard dict if necessary
            config_dict = _to_dict(config)

            try:
                validated_toolkit = validate_toolkit_config(config_dict)
                toolkit_name = validated_toolkit.name
                logger.info(f"✅ Toolkit configuration validated: {toolkit_name}")
            except Exception as e:
                logger.error(f"Invalid toolkit configuration: {e}")
                raise ValueError(f"Toolkit validation failed: {e}") from e

            if toolkit_name in self._toolkits:
                try:
                    toolkit_class = self._toolkits[toolkit_name]
                    params = validated_toolkit.params
                    param_class = ToolkitConfig.get_toolkit_params_class(toolkit_name)
                    if param_class and 'validate_credentials' in param_class.model_fields:
                        has_api_key_fields = any(
                            field_name in ['api_key', 'api_secret']
                            for field_name in param_class.model_fields
                        )
                        if has_api_key_fields:
                            params["validate_credentials"] = True
                            logger.debug(f"Enabled credential validation for {toolkit_name} (has API key fields)")
                            try:
                                param_class(**params)
                                logger.debug(f"✅ API credentials validated for {toolkit_name}")
                            except Exception as validation_error:
                                raise validation_error
                        else:
                            logger.debug(f"Skipping credential validation for {toolkit_name} (no API key fields)")

                    params = self._transform_toolkit_params(toolkit_name, params)
                    toolkit_params = {k: v for k, v in params.items() if k != 'validate_credentials'}
                    available_tools = validated_toolkit.available_tools
                    if available_tools:
                        toolkit_params["include_tools"] = available_tools

                    toolkit_instance = toolkit_class(**toolkit_params)
                    logger.info(f"Created custom toolkit '{toolkit_name}' with params: {params}")
                    selected_toolkits.append(toolkit_instance)

                except Exception as e:
                    logger.warning(f"Custom toolkit '{toolkit_name}' failed to load - skipping: {e}")
                    logger.info(f"💡 Check {toolkit_name} configuration and dependencies")
            else:
                try:
                    agno_toolkit_instance = self._create_agno_toolkit(toolkit_name, validated_toolkit)
                    if agno_toolkit_instance:
                        selected_toolkits.append(agno_toolkit_instance)
                        logger.info(f"Created agno toolkit '{toolkit_name}'")
                    else:
                        available_custom = list(self._toolkits.keys())
                        logger.warning(f"Toolkit '{toolkit_name}' not available - skipping")
                        logger.info(f"💡 Available custom toolkits: {available_custom}")
                        logger.info(f"💡 Please check default Agno toolkits for using pre-defined tools")
                except Exception as e:
                    logger.warning(f"Toolkit '{toolkit_name} with params {validated_toolkit.params}' failed to load - skipping: {e}")
                    logger.info(f"💡 If you need {toolkit_name}, check that all dependencies are installed and configured")

        return selected_toolkits

    def _transform_toolkit_params(self, toolkit_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform toolkit parameters if needed."""
        if toolkit_name in self._toolkits:
            data_dir = params.get("data_dir")
            if data_dir and not os.path.isabs(data_dir):
                project_root = Path(__file__).resolve().parents[4]
                params["data_dir"] = str(project_root / data_dir)
                logger.debug(f"Converted relative data_dir to absolute: {params['data_dir']}")
        return params

    def _create_agno_toolkit(self, toolkit_name: str, validated_toolkit) -> Optional[Any]:
        """Dynamically create an agno toolkit instance."""
        try:
            module_path = f"agno.tools.{toolkit_name.lower().replace('tools', '')}"
            if module_path.endswith('.'):
                module_path = module_path[:-1]

            logger.debug(f"Attempting to import agno toolkit from: {module_path}")
            toolkit_class = None
            try:
                module = importlib.import_module(module_path)
                toolkit_class = getattr(module, toolkit_name, None)
            except ImportError:
                common_locations = [f"agno.tools.{toolkit_name.lower()}", "agno.tools", f"agno.{toolkit_name.lower()}"]
                for location in common_locations:
                    try:
                        module = importlib.import_module(location)
                        toolkit_class = getattr(module, toolkit_name, None)
                        if toolkit_class:
                            logger.debug(f"Found {toolkit_name} in {location}")
                            break
                    except (ImportError, AttributeError):
                        continue

            if not toolkit_class:
                logger.warning(f"Agno toolkit class {toolkit_name} not found in any expected location")
                return None

            params = _to_dict(validated_toolkit.params or {}) # Convert OmegaConf params

            if toolkit_name == "E2BTools":
                if "timeout" not in params:
                    timeout = int(os.getenv("E2B_TIMEOUT", "300"))
                    params["timeout"] = timeout
                    logger.debug(f"Set E2B timeout to: {timeout}s")
                if "sandbox_options" not in params:
                    params["sandbox_options"] = {}
                if "template" not in params["sandbox_options"]:
                    template_id = os.getenv("E2B_TEMPLATE_ID", "sentient-e2b-s3")
                    params["sandbox_options"]["template"] = template_id
                    logger.debug(f"Set E2B template to: {template_id}")

            toolkit_instance = toolkit_class(**params)
            logger.debug(f"Created agno toolkit {toolkit_name} with params: {params}")
            return toolkit_instance

        except Exception as e:
            import traceback
            logger.error(f"Failed to create agno toolkit {toolkit_name} with params {validated_toolkit.params}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def create_agno_agent(self, agent_config: Union[DictConfig, AgentConfig]) -> Optional[AgnoAgent]:
        """
        Create an AgnoAgent instance from configuration with proper structured output support.

        Args:
            agent_config: Agent configuration (DictConfig or AgentConfig)

        Returns:
            AgnoAgent instance or None for agents that don't use AgnoAgent
        """
        # Convert OmegaConf to standard dict if needed, handle AgentConfig directly
        if isinstance(agent_config, AgentConfig):
             config_dict = agent_config.model_dump(exclude_none=True) # Use Pydantic's method
        else:
             config_dict = _to_dict(agent_config) # Convert OmegaConf


        agent_name = config_dict.get("name", "UnknownAgent")
        adapter_class_name = config_dict.get("adapter_class", "")

        if adapter_class_name in ["OpenAICustomSearchAdapter", "GeminiCustomSearchAdapter", "ExaCustomSearchAdapter"]:
            logger.debug(f"Agent {agent_name} doesn't use AgnoAgent (custom search adapter)")
            return None

        if "model" not in config_dict or "prompt_source" not in config_dict:
            logger.debug(f"Agent {agent_name} doesn't use AgnoAgent (no model/prompt_source)")
            return None

        try:
            # Prepare model configuration
            model_config_data = config_dict["model"]

            # --- *** ADD JSON MODE LOGIC HERE *** ---
            response_model_name = config_dict.get("response_model")
            if response_model_name:
                # Check if the provider supports JSON mode (currently litellm via OpenAI protocol)
                provider = model_config_data.get("provider", "").lower()
                # Check if we are using local OpenAI endpoint or standard OpenAI
                is_openai_compatible = (provider == "litellm" and os.getenv("OPENAI_API_BASE_URL")) or provider == "openai"

                if is_openai_compatible:
                    # Inject response_format into the model config dict
                    model_config_data["response_format"] = {"type": "json_object"}
                    logger.info(f"💡 Enabled JSON mode (response_format) for agent '{agent_name}' due to response_model '{response_model_name}'.")
                else:
                    logger.warning(f"Agent '{agent_name}' has response_model '{response_model_name}' but provider '{provider}' might not support JSON mode via response_format.")
            # --- *** END JSON MODE LOGIC *** ---


            # Create model instance using the potentially modified config
            model = self.create_model_instance(model_config_data)
            logger.debug(f"Created model for {agent_name}: {model_config_data.get('provider')}/{model_config_data.get('model_id')}")

            # Resolve system prompt
            system_message = self.config_loader.resolve_prompt(config_dict["prompt_source"])
            logger.debug(f"Resolved prompt for {agent_name} from {config_dict['prompt_source']}")

            # Get response model class if specified
            response_model = None
            if response_model_name:
                response_model = self.resolve_response_model(response_model_name)
                if response_model:
                    logger.debug(f"Using response model for {agent_name}: {response_model_name}")
                else:
                    logger.warning(f"Could not resolve response model {response_model_name} for {agent_name}")

            # Create tools if specified
            tools = []
            if "tools" in config_dict and config_dict["tools"]:
                tools.extend(self.create_tools(config_dict["tools"]))

            if "toolkits" in config_dict and config_dict["toolkits"]:
                tools.extend(self.create_toolkits(config_dict["toolkits"]))

            if tools:
                logger.debug(f"Created {len(tools)} tools/toolkit functions for {agent_name}")

            # Prepare AgnoAgent kwargs
            agno_kwargs = {
                "model": model,
                "system_message": system_message,
                "name": f"{agent_name}_Agno",
            }

            if response_model:
                agno_kwargs["response_model"] = response_model

            if tools:
                agno_kwargs["tools"] = tools

            # Handle additional AgnoAgent parameters
            if "agno_params" in config_dict and config_dict["agno_params"] is not None:
                additional_params = config_dict["agno_params"]
                agno_kwargs.update(additional_params)
                logger.debug(f"Added additional AgnoAgent params for {agent_name}: {list(additional_params.keys())}")

            logger.debug(f"Creating AgnoAgent for {agent_name} with kwargs: {list(agno_kwargs.keys())}")

            agno_agent = AgnoAgent(**agno_kwargs)
            logger.info(f"✅ Created AgnoAgent for {agent_name}")
            return agno_agent

        except Exception as e:
            logger.error(f"❌ Failed to create AgnoAgent for {agent_name}: {e}")
            raise

    def create_adapter(self, agent_config: Union[DictConfig, AgentConfig], agno_agent: Optional[AgnoAgent] = None) -> Any:
        """
        Create an adapter instance from configuration.

        Args:
            agent_config: Agent configuration (DictConfig or AgentConfig)
            agno_agent: Optional AgnoAgent instance

        Returns:
            Adapter instance
        """
        # Convert OmegaConf to standard dict if needed, handle AgentConfig directly
        if isinstance(agent_config, AgentConfig):
             config_dict = agent_config.model_dump(exclude_none=True)
        else:
             config_dict = _to_dict(agent_config)

        agent_name = config_dict.get("name", "UnknownAgent")
        adapter_class_name = config_dict["adapter_class"]

        if adapter_class_name not in self._adapter_classes:
            raise ValueError(f"Unknown adapter class: {adapter_class_name}")

        adapter_class = self._adapter_classes[adapter_class_name]

        try:
            if adapter_class_name in ["OpenAICustomSearchAdapter", "GeminiCustomSearchAdapter", "ExaCustomSearchAdapter"]:
                adapter_kwargs = {}
                if "adapter_params" in config_dict and config_dict["adapter_params"] is not None:
                    adapter_kwargs.update(config_dict["adapter_params"])
                if "model" in config_dict and config_dict["model"] is not None and "model_id" in config_dict["model"]:
                    adapter_kwargs["model_id"] = config_dict["model"]["model_id"]
                return adapter_class(**adapter_kwargs)
            else:
                if agno_agent is None:
                    raise ValueError(f"Adapter {adapter_class_name} requires an AgnoAgent instance")
                adapter_kwargs = {"agno_agent_instance": agno_agent, "agent_name": agent_name}
                if "adapter_params" in config_dict and config_dict["adapter_params"] is not None:
                    additional_params = config_dict["adapter_params"]
                    adapter_kwargs.update(additional_params)
                    logger.debug(f"Added additional adapter params for {agent_name}: {list(additional_params.keys())}")
                return adapter_class(**adapter_kwargs)

        except Exception as e:
            logger.error(f"❌ Failed to create adapter {adapter_class_name} for {agent_name}: {e}")
            raise

    def create_agent(self, agent_config: Union[DictConfig, Dict[str, Any], AgentConfig]) -> Dict[str, Any]:
        """
        Create a complete agent (AgnoAgent + Adapter) from configuration with Pydantic validation.

        Args:
            agent_config: Agent configuration (DictConfig, dict, or AgentConfig)

        Returns:
            Dictionary containing agent components and metadata
        """
        if isinstance(agent_config, AgentConfig):
            validated_config = agent_config
            # Convert validated Pydantic model back to dict for processing
            config_dict = validated_config.model_dump(exclude_none=True)
        else:
             # Convert OmegaConf or dict to standard dict first
             config_dict = _to_dict(agent_config)
             try:
                validated_config = validate_agent_config(config_dict)
                logger.debug(f"✅ Agent configuration validated: {validated_config.name}")
             except Exception as e:
                logger.error(f"Agent configuration validation failed: {e}")
                raise ValueError(f"Invalid agent configuration: {e}") from e


        agent_name = validated_config.name
        logger.info(f"🔧 Creating agent: {agent_name} (type: {validated_config.type})")

        try:
            # Pass the validated AgentConfig object or converted dict
            agno_agent = self.create_agno_agent(validated_config)
            adapter = self.create_adapter(validated_config, agno_agent)


            if not isinstance(adapter, BaseAdapter):
                logger.error(f"❌ Created adapter for {agent_name} is not a BaseAdapter!")
                raise TypeError(f"Adapter {type(adapter)} is not a BaseAdapter")

            logger.info(f"✅ Created valid BaseAdapter for {agent_name}: {type(adapter).__name__}")

            registration_info = {"action_keys": [], "named_keys": []}
            if validated_config.registration:
                reg_config = validated_config.registration
                if reg_config.action_keys:
                    action_keys = []
                    for key in reg_config.action_keys:
                        action_verb = key.action_verb
                        task_type_value = key.task_type
                        if task_type_value is not None and isinstance(task_type_value, str):
                            try:
                                task_type_enum = TaskType[task_type_value.upper()]
                                action_keys.append((action_verb, task_type_enum))
                            except KeyError:
                                logger.error(f"Invalid task_type '{task_type_value}' for agent {agent_name}. Valid: {list(TaskType.__members__.keys())}")
                                raise ValueError(f"Invalid task_type: {task_type_value}")
                        else:
                             action_keys.append((action_verb, task_type_value)) # Handle None or already enum
                    registration_info["action_keys"] = action_keys

                if reg_config.named_keys:
                    registration_info["named_keys"] = list(reg_config.named_keys)


            runtime_tools = []
            if agno_agent and hasattr(agno_agent, 'tools') and agno_agent.tools:
                 runtime_tools = [{"name": getattr(tool, '__name__', str(type(tool).__name__)), "params": {}} for tool in agno_agent.tools]

            metadata = {
                "has_structured_output": bool(validated_config.response_model),
                "response_model": validated_config.response_model,
                "has_tools": bool(runtime_tools) or bool(validated_config.tools) or bool(validated_config.toolkits),
                "tools": runtime_tools, # Show runtime tools primarily
                "model_info": validated_config.model.model_dump(exclude_none=True) if validated_config.model else None,
                "prompt_source": validated_config.prompt_source,
            }

            agent_info = {
                "name": agent_name,
                "type": validated_config.type,
                "description": validated_config.description or "",
                "adapter": adapter,
                "agno_agent": agno_agent,
                "registration": registration_info,
                "enabled": validated_config.enabled,
                "metadata": metadata,
                "config": config_dict # Store original dict representation
            }

            logger.info(f"✅ Successfully created agent: {agent_name}")
            if metadata["has_structured_output"]:
                logger.info(f"   📋 Structured output: {metadata['response_model']}")
            if metadata["has_tools"]:
                 tool_names = [t['name'] for t in metadata['tools']]
                 logger.info(f"   🔧 Tools: {tool_names}")


            return agent_info

        except Exception as e:
            logger.error(f"❌ Failed to create agent {agent_name}: {e}")
            raise

    def create_all_agents(self, config: AgentsYAMLConfig) -> Dict[str, Dict[str, Any]]:
        """
        Create all agents from configuration.

        Args:
            config: Validated AgentsYAMLConfig object

        Returns:
            Dictionary mapping agent names to agent info
        """
        agents = {}
        created_count = 0
        skipped_count = 0
        failed_count = 0

        logger.info(f"🚀 Creating {len(config.agents)} agents from configuration...")

        for agent_config in config.agents: # Iterate over Pydantic AgentConfig objects
            try:
                if not agent_config.enabled:
                    logger.info(f"⏭️  Skipping disabled agent: {agent_config.name}")
                    skipped_count += 1
                    continue

                agent_info = self.create_agent(agent_config) # Pass the AgentConfig object directly
                agents[agent_config.name] = agent_info
                created_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to create agent {agent_config.name}: {e}")
                failed_count += 1
                continue

        logger.info(f"📊 Agent creation summary:")
        logger.info(f"   ✅ Created: {created_count}")
        logger.info(f"   ⏭️  Skipped: {skipped_count}")
        logger.info(f"   ❌ Failed: {failed_count}")

        return agents

    def create_agents_for_profile(self, profile_config: 'DictConfig') -> Dict[str, Dict[str, Any]]:
        """
        Create agents specifically for a profile configuration.

        Args:
            profile_config: Profile configuration from YAML

        Returns:
            Dictionary of created agents
        """
        created_agents = {}
        if "agents" in profile_config and profile_config.agents:
            logger.info(f"Creating {len(profile_config.agents)} profile-specific agents...")
            for agent_config_data in profile_config.agents:
                # Convert OmegaConf DictConfig to standard dict before validation/creation
                agent_config_dict = _to_dict(agent_config_data)
                if agent_config_dict.get("enabled", True):
                    try:
                        agent_info = self.create_agent(agent_config_dict)
                        created_agents[agent_config_dict["name"]] = agent_info
                        logger.info(f"✅ Created profile agent: {agent_config_dict['name']}")
                    except Exception as e:
                        logger.error(f"❌ Failed to create profile agent {agent_config_dict['name']}: {e}")
        return created_agents

    def validate_blueprint_agents(self, blueprint: 'AgentBlueprint', agent_registry: AgentRegistry) -> Dict[str, Any]:
        """
        Validate that all agents referenced in a blueprint exist in the given registry instance.

        Args:
            blueprint: AgentBlueprint to validate.
            agent_registry: The AgentRegistry instance to check against.

        Returns:
            Validation results.
        """
        validation = {
            "valid": True,
            "missing_agents": [],
            "issues": [],
            "available_agents": list(agent_registry.get_all_named_agents().keys()),
            "blueprint_agents": []
        }
        named_agents = agent_registry.get_all_named_agents()

        for task_type, planner_name in blueprint.planner_adapter_names.items():
            validation["blueprint_agents"].append(f"Planner({task_type.value}): {planner_name}")
            if planner_name not in named_agents:
                validation["missing_agents"].append(f"Planner: {planner_name}")
                validation["valid"] = False

        for task_type, executor_name in blueprint.executor_adapter_names.items():
            validation["blueprint_agents"].append(f"Executor({task_type.value}): {executor_name}")
            if executor_name not in named_agents:
                validation["missing_agents"].append(f"Executor: {executor_name}")
                validation["valid"] = False

        other_agents = [
            ("Root Planner", blueprint.root_planner_adapter_name),
            ("Aggregator", blueprint.aggregator_adapter_name),
            ("Atomizer", blueprint.atomizer_adapter_name),
            ("PlanModifier", blueprint.plan_modifier_adapter_name),
            ("Default Planner", blueprint.default_planner_adapter_name),
            ("Default Executor", blueprint.default_executor_adapter_name),
        ]

        for agent_type, agent_name in other_agents:
            if agent_name:
                validation["blueprint_agents"].append(f"{agent_type}: {agent_name}")
                if agent_name not in named_agents:
                    validation["missing_agents"].append(f"{agent_type}: {agent_name}")
                    validation["valid"] = False

        if validation["missing_agents"]:
            validation["issues"].extend([
                f"Agent '{name}' referenced in blueprint '{blueprint.name}' is not registered."
                for name in validation["missing_agents"]
            ])

        return validation


def create_agents_from_config(config: AgentsYAMLConfig, config_loader) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to create all agents from validated configuration.

    Args:
        config: Validated AgentsYAMLConfig object
        config_loader: AgentConfigLoader instance

    Returns:
        Dictionary of created agents
    """
    factory = AgentFactory(config_loader)
    return factory.create_all_agents(config)

