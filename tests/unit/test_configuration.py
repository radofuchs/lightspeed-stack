"""Unit tests for functions defined in src/configuration.py."""

# pylint: disable=too-many-lines

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import constants
from cache.in_memory_cache import InMemoryCache
from cache.sqlite_cache import SQLiteCache
from configuration import AppConfig, LogicError
from models.config import CustomProfile, ModelContextProtocolServer


# pylint: disable=broad-exception-caught,protected-access
@pytest.fixture(autouse=True)
def _reset_app_config_between_tests() -> Generator:
    # ensure clean state before each test
    """
    Reset AppConfig singleton internal state before and after a test to avoid test contamination.

    Attempts to set AppConfig()._configuration to None and
    AppConfig()._quota_limiters to an empty list, ignoring any exceptions, then
    yields control to the test and repeats the cleanup after the test.
    """
    try:
        AppConfig()._configuration = None  # type: ignore[attr-defined]
        AppConfig()._quota_limiters = []  # type: ignore[attr-defined]
    except Exception:
        pass
    yield
    # ensure clean state after each test
    try:
        AppConfig()._configuration = None  # type: ignore[attr-defined]
        AppConfig()._quota_limiters = []  # type: ignore[attr-defined]
    except Exception:
        pass


def test_default_configuration() -> None:
    """Test that configuration attributes are not accessible for uninitialized app."""
    cfg = AppConfig()
    assert cfg is not None

    # configuration is not loaded
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.service_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.llama_stack_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = (
            cfg.user_data_collection_configuration
        )  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.mcp_servers  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.authentication_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.customization  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.authorization_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.inference  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.database_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.conversation_cache_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.quota_handlers_configuration  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.conversation_cache  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.quota_limiters  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.a2a_state  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.token_usage_history  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.azure_entra_id  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.splunk  # pylint: disable=pointless-statement

    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        # try to read property
        _ = cfg.deployment_environment  # pylint: disable=pointless-statement


def test_configuration_is_singleton() -> None:
    """Test that configuration is singleton."""
    cfg1 = AppConfig()
    cfg2 = AppConfig()
    assert cfg1 == cfg2


def test_init_from_dict() -> None:
    """Test the configuration initialization from dictionary with config values."""
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "mcp_servers": [],
        "customization": None,
        "authentication": {
            "module": "noop",
        },
        "a2a_state": {
            "sqlite": None,
            "postgres": None,
        },
        "splunk": {
            "enabled": False,
            "url": "foo.bar.baz",
            "index": "index",
            "source": "source",
            "timeout": 10,
            "verify_ssl": False,
        },
        "deployment_environment": "foo",
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    # check for all subsections
    assert cfg.configuration is not None
    assert cfg.llama_stack_configuration is not None
    assert cfg.service_configuration is not None
    assert cfg.user_data_collection_configuration is not None

    # check for configuration subsection
    assert cfg.configuration.name == "foo"

    # check for llama_stack_configuration subsection
    assert cfg.llama_stack_configuration.api_key is not None
    assert cfg.llama_stack_configuration.api_key.get_secret_value() == "xyzzy"
    assert str(cfg.llama_stack_configuration.url) == "http://x.y.com:1234/"
    assert cfg.llama_stack_configuration.use_as_library_client is False

    # check for service_configuration subsection
    assert cfg.service_configuration.host == "localhost"
    assert cfg.service_configuration.port == 8080
    assert cfg.service_configuration.auth_enabled is False
    assert cfg.service_configuration.workers == 1
    assert cfg.service_configuration.color_log is True
    assert cfg.service_configuration.access_log is True

    # check for user data collection subsection
    assert cfg.user_data_collection_configuration.feedback_enabled is False

    # check authentication_configuration
    assert cfg.authentication_configuration is not None
    assert cfg.authentication_configuration.module == "noop"

    # check authorization configuration - default value
    assert cfg.authorization_configuration is not None

    # check database configuration
    assert cfg.database_configuration is not None

    # check inference configuration
    assert cfg.inference is not None

    # check conversation cache
    assert cfg.conversation_cache_configuration is not None

    # check a2a state
    assert cfg.a2a_state is not None
    assert cfg.a2a_state.sqlite is None
    assert cfg.a2a_state.postgres is None

    # check Splunk
    assert cfg.splunk is not None
    assert cfg.splunk.enabled is False
    assert cfg.splunk.url == "foo.bar.baz"
    assert cfg.splunk.index == "index"
    assert cfg.splunk.source == "source"
    assert cfg.splunk.timeout == 10
    assert cfg.splunk.verify_ssl is False

    # check deployment_environment
    assert cfg.deployment_environment is not None

    # check token usage history
    assert cfg.token_usage_history is None


def test_init_from_dict_with_mcp_servers() -> None:
    """Test initialization with MCP servers configuration."""
    config_dict = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "mcp_servers": [
            {
                "name": "server1",
                "url": "http://localhost:8080",
            },
            {
                "name": "server2",
                "provider_id": "custom-provider",
                "url": "https://api.example.com",
            },
        ],
        "customization": None,
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    assert len(cfg.mcp_servers) == 2
    assert cfg.mcp_servers[0].name == "server1"
    assert cfg.mcp_servers[0].provider_id == "model-context-protocol"
    assert cfg.mcp_servers[0].url == "http://localhost:8080"
    assert cfg.mcp_servers[1].name == "server2"
    assert cfg.mcp_servers[1].provider_id == "custom-provider"
    assert cfg.mcp_servers[1].url == "https://api.example.com"


def test_init_from_dict_with_authorization_configuration() -> None:
    """Test initialization with authorization configuration configuration.

    Verify AppConfig initializes authorization configuration when an empty
    `authorization` block is provided.

    Initializes the singleton AppConfig from a dict that includes an empty
    `authorization` section and asserts that `authorization_configuration` is
    not None.
    """
    config_dict = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "authorization": {},
        "customization": None,
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    assert cfg.authorization_configuration is not None


def test_load_proper_configuration(tmpdir: Path) -> None:
    """Test loading proper configuration from YAML file.

    Verify that a valid YAML configuration file loads and populates key AppConfig sections.

    Writes a YAML configuration to a temporary file, loads it with
    AppConfig.load_configuration, and asserts that `configuration`,
    `llama_stack_configuration`, `service_configuration`, and
    `user_data_collection_configuration` are populated.
    """
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: foo bar baz
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: xyzzy
user_data_collection:
  feedback_enabled: false
mcp_servers: []
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))
    assert cfg.configuration is not None
    assert cfg.llama_stack_configuration is not None
    assert cfg.service_configuration is not None
    assert cfg.user_data_collection_configuration is not None


def test_load_configuration_with_mcp_servers(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with MCP servers."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
mcp_servers:
  - name: filesystem-server
    url: http://localhost:3000
  - name: git-server
    provider_id: custom-git-provider
    url: https://git.example.com/mcp
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert len(cfg.mcp_servers) == 2
    assert cfg.mcp_servers[0].name == "filesystem-server"
    assert cfg.mcp_servers[0].provider_id == "model-context-protocol"
    assert cfg.mcp_servers[0].url == "http://localhost:3000"
    assert cfg.mcp_servers[1].name == "git-server"
    assert cfg.mcp_servers[1].provider_id == "custom-git-provider"
    assert cfg.mcp_servers[1].url == "https://git.example.com/mcp"


def test_mcp_servers_property_empty() -> None:
    """Test mcp_servers property returns empty list when no servers configured."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "test-key",
            "url": "http://localhost:8321",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "mcp_servers": [],
        "customization": None,
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    servers = cfg.mcp_servers
    assert isinstance(servers, list)
    assert len(servers) == 0


def test_mcp_servers_property_with_servers() -> None:
    """Test mcp_servers property returns correct list of ModelContextProtocolServer objects."""
    config_dict = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "test-key",
            "url": "http://localhost:8321",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "mcp_servers": [
            {
                "name": "test-server",
                "url": "http://localhost:8080",
            },
        ],
        "customization": None,
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    servers = cfg.mcp_servers
    assert isinstance(servers, list)
    assert len(servers) == 1
    assert isinstance(servers[0], ModelContextProtocolServer)
    assert servers[0].name == "test-server"
    assert servers[0].url == "http://localhost:8080"


def test_configuration_not_loaded() -> None:
    """Test that accessing configuration before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.configuration
        assert c is not None


def test_service_configuration_not_loaded() -> None:
    """Test that accessing service_configuration before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.service_configuration
        assert c is not None


def test_llama_stack_configuration_not_loaded() -> None:
    """Test that accessing llama_stack_configuration before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.llama_stack_configuration
        assert c is not None


def test_user_data_collection_configuration_not_loaded() -> None:
    """Test that accessing user_data_collection_configuration before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.user_data_collection_configuration
        assert c is not None


def test_mcp_servers_not_loaded() -> None:
    """Test that accessing mcp_servers before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.mcp_servers
        assert c is not None


def test_authentication_configuration_not_loaded() -> None:
    """Test that accessing authentication_configuration before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.authentication_configuration
        assert c is not None


def test_customization_not_loaded() -> None:
    """Test that accessing customization before loading raises an error."""
    cfg = AppConfig()
    with pytest.raises(LogicError, match="logic error: configuration is not loaded"):
        c = cfg.customization
        assert c is not None


def test_load_configuration_with_customization_system_prompt_path(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with customization."""
    system_prompt_filename = tmpdir / "system_prompt.txt"
    with open(system_prompt_filename, "w", encoding="utf-8") as fout:
        fout.write("this is system prompt")

    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write(f"""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
mcp_servers:
  - name: filesystem-server
    url: http://localhost:3000
  - name: git-server
    provider_id: custom-git-provider
    url: https://git.example.com/mcp
customization:
  disable_query_system_prompt: true
  system_prompt_path: {system_prompt_filename}
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.customization is not None
    assert cfg.customization.system_prompt is not None
    assert cfg.customization.system_prompt == "this is system prompt"


def test_load_configuration_with_customization_system_prompt(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with system_prompt in the customization."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
mcp_servers:
  - name: filesystem-server
    url: http://localhost:3000
  - name: git-server
    provider_id: custom-git-provider
    url: https://git.example.com/mcp
customization:
  system_prompt: |-
    this is system prompt in the customization section
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.customization is not None
    assert cfg.customization.system_prompt is not None
    assert (
        cfg.customization.system_prompt.strip()
        == "this is system prompt in the customization section"
    )


def test_configuration_with_profile_customization(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with a custom profile."""
    expected_profile = CustomProfile(path="tests/profiles/test/profile.py")
    expected_prompts = expected_profile.get_prompts()
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
customization:
  profile_path: tests/profiles/test/profile.py
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert (
        cfg.customization is not None and cfg.customization.custom_profile is not None
    )
    fetched_prompts = cfg.customization.custom_profile.get_prompts()
    assert fetched_prompts is not None and fetched_prompts.get(
        "default"
    ) == expected_prompts.get("default")


def test_configuration_with_all_customizations(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with a custom profile, prompt and prompt path."""
    expected_profile = CustomProfile(path="tests/profiles/test/profile.py")
    expected_prompts = expected_profile.get_prompts()
    system_prompt_filename = tmpdir / "system_prompt.txt"
    with open(system_prompt_filename, "w", encoding="utf-8") as fout:
        fout.write("this is system prompt")

    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write(f"""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
customization:
  profile_path: tests/profiles/test/profile.py
  system_prompt: custom prompt
  system_prompt_path: {system_prompt_filename}
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert (
        cfg.customization is not None and cfg.customization.custom_profile is not None
    )
    fetched_prompts = cfg.customization.custom_profile.get_prompts()
    assert fetched_prompts is not None and fetched_prompts.get(
        "default"
    ) == expected_prompts.get("default")


def test_configuration_with_sqlite_conversation_cache(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with conversation cache configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
conversation_cache:
  type: "sqlite"
  sqlite:
    db_path: ":memory:"
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.conversation_cache_configuration is not None
    assert cfg.conversation_cache_configuration.type == "sqlite"
    assert cfg.conversation_cache_configuration.sqlite is not None
    assert cfg.conversation_cache_configuration.postgres is None
    assert cfg.conversation_cache_configuration.memory is None
    assert cfg.conversation_cache is not None
    assert isinstance(cfg.conversation_cache, SQLiteCache)


def test_configuration_with_in_memory_conversation_cache(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with conversation cache configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
conversation_cache:
  type: "memory"
  memory:
    max_entries: 42
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.conversation_cache_configuration is not None
    assert cfg.conversation_cache_configuration.type == "memory"
    assert cfg.conversation_cache_configuration.sqlite is None
    assert cfg.conversation_cache_configuration.postgres is None
    assert cfg.conversation_cache_configuration.memory is not None
    assert cfg.conversation_cache is not None
    assert isinstance(cfg.conversation_cache, InMemoryCache)


def test_configuration_with_quota_handlers_no_storage(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with quota handlers configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
quota_handlers:
  limiters:
    - name: user_monthly_limits
      type: user_limiter
      initial_quota: 10
      quota_increase: 10
      period: "2 seconds"
    - name: cluster_monthly_limits
      type: cluster_limiter
      initial_quota: 100
      quota_increase: 10
      period: "10 seconds"
  scheduler:
    # scheduler ticks in seconds
    period: 1
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.quota_handlers_configuration is not None
    assert cfg.quota_handlers_configuration.sqlite is None
    assert cfg.quota_handlers_configuration.postgres is None
    assert cfg.quota_handlers_configuration.limiters is not None
    assert cfg.quota_handlers_configuration.scheduler is not None

    # check the quota limiters configuration
    assert len(cfg.quota_limiters) == 0

    # check the scheduler configuration
    assert cfg.quota_handlers_configuration.scheduler.period == 1


def test_configuration_with_token_history_no_storage(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with quota handlers configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
quota_handlers:
  scheduler:
    # scheduler ticks in seconds
    period: 1
  enable_token_history: true
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.quota_handlers_configuration is not None
    assert cfg.quota_handlers_configuration.sqlite is None
    assert cfg.quota_handlers_configuration.postgres is None
    assert cfg.quota_handlers_configuration.scheduler is not None

    # check the token usage history
    assert cfg.token_usage_history is not None


def test_configuration_with_quota_handlers(tmpdir: Path) -> None:
    """Test loading configuration from YAML file with quota handlers configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: false
  url: http://localhost:8321
  api_key: test-key
user_data_collection:
  feedback_enabled: false
quota_handlers:
  sqlite:
      db_path: ":memory:"
  limiters:
    - name: user_monthly_limits
      type: user_limiter
      initial_quota: 10
      quota_increase: 10
      period: "2 seconds"
    - name: cluster_monthly_limits
      type: cluster_limiter
      initial_quota: 100
      quota_increase: 10
      period: "10 seconds"
  scheduler:
    # scheduler ticks in seconds
    period: 1
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    assert cfg.quota_handlers_configuration is not None
    assert cfg.quota_handlers_configuration.sqlite is not None
    assert cfg.quota_handlers_configuration.postgres is None
    assert cfg.quota_handlers_configuration.limiters is not None
    assert cfg.quota_handlers_configuration.scheduler is not None

    # check the storage
    assert cfg.quota_handlers_configuration.sqlite.db_path == ":memory:"

    # check the quota limiters configuration
    assert len(cfg.quota_limiters) == 2
    assert (
        str(cfg.quota_limiters[0])
        == "UserQuotaLimiter: initial quota: 10 increase by: 10"
    )
    assert (
        str(cfg.quota_limiters[1])
        == "ClusterQuotaLimiter: initial quota: 100 increase by: 10"
    )

    # check the scheduler configuration
    assert cfg.quota_handlers_configuration.scheduler.period == 1


def test_load_configuration_with_azure_entra_id(tmpdir: Path) -> None:
    """Return Azure Entra ID configuration when provided in configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  api_key: test-key
  url: http://localhost:8321
  use_as_library_client: false
user_data_collection:
  feedback_enabled: false
azure_entra_id:
  tenant_id: tenant
  client_id: client
  client_secret: secret
            """)

    cfg = AppConfig()
    cfg.load_configuration(str(cfg_filename))

    azure_conf = cfg.azure_entra_id
    assert azure_conf is not None
    assert azure_conf.tenant_id.get_secret_value() == "tenant"
    assert azure_conf.client_id.get_secret_value() == "client"
    assert azure_conf.client_secret.get_secret_value() == "secret"


def test_load_configuration_with_incomplete_azure_entra_id_raises(tmpdir: Path) -> None:
    """Raise error if Azure Entra ID block is incomplete in configuration."""
    cfg_filename = tmpdir / "config.yaml"
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        fout.write("""
name: test service
service:
  host: localhost
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  api_key: test-key
  url: http://localhost:8321
  use_as_library_client: false
user_data_collection:
  feedback_enabled: false
azure_entra_id:
  tenant_id: tenant
  client_id: client
            """)

    cfg = AppConfig()
    with pytest.raises(ValidationError):
        cfg.load_configuration(str(cfg_filename))


def test_rag_id_mapping_excludes_solr_when_okp_not_configured(
    minimal_config: AppConfig,
) -> None:
    """Test that rag_id_mapping does not include OKP/Solr when OKP is not in rag config."""
    assert minimal_config.rag_id_mapping == {}


def test_rag_id_mapping_includes_solr_when_okp_in_inline() -> None:
    """Test that rag_id_mapping includes OKP/Solr mapping when OKP is in rag.inline."""
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "rag": {"inline": [constants.OKP_RAG_ID]},
        }
    )
    assert constants.SOLR_DEFAULT_VECTOR_STORE_ID in cfg.rag_id_mapping
    assert (
        cfg.rag_id_mapping[constants.SOLR_DEFAULT_VECTOR_STORE_ID]
        == constants.OKP_RAG_ID
    )


def test_rag_id_mapping_includes_solr_when_okp_in_tool() -> None:
    """Test that rag_id_mapping includes OKP/Solr mapping when OKP is in rag.tool."""
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "rag": {"tool": [constants.OKP_RAG_ID]},
        }
    )
    assert constants.SOLR_DEFAULT_VECTOR_STORE_ID in cfg.rag_id_mapping
    assert (
        cfg.rag_id_mapping[constants.SOLR_DEFAULT_VECTOR_STORE_ID]
        == constants.OKP_RAG_ID
    )


def test_rag_id_mapping_with_byok(tmp_path: Path) -> None:
    """Test that rag_id_mapping builds correct mapping from BYOK config."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "byok_rag": [
                {
                    "rag_id": "my-kb",
                    "vector_db_id": "vs-001",
                    "db_path": str(db_file),
                },
            ],
        }
    )
    assert cfg.rag_id_mapping == {"vs-001": "my-kb"}


def test_rag_id_mapping_with_byok_and_okp(tmp_path: Path) -> None:
    """Test that rag_id_mapping includes both BYOK and OKP entries when OKP is configured."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "rag": {"inline": [constants.OKP_RAG_ID]},
            "byok_rag": [
                {
                    "rag_id": "my-kb",
                    "vector_db_id": "vs-001",
                    "db_path": str(db_file),
                },
            ],
        }
    )
    assert "vs-001" in cfg.rag_id_mapping
    assert cfg.rag_id_mapping["vs-001"] == "my-kb"
    assert constants.SOLR_DEFAULT_VECTOR_STORE_ID in cfg.rag_id_mapping
    assert (
        cfg.rag_id_mapping[constants.SOLR_DEFAULT_VECTOR_STORE_ID]
        == constants.OKP_RAG_ID
    )


def test_resolve_index_name_with_mapping(minimal_config: AppConfig) -> None:
    """Test resolve_index_name uses mapping when available."""
    mapping = {"vs-x": "user-friendly-name"}
    assert minimal_config.resolve_index_name("vs-x", mapping) == "user-friendly-name"


def test_resolve_index_name_passthrough(minimal_config: AppConfig) -> None:
    """Test resolve_index_name passes through unmapped IDs."""
    assert minimal_config.resolve_index_name("vs-unknown", {}) == "vs-unknown"


def test_rag_id_mapping_not_loaded() -> None:
    """Test that rag_id_mapping raises when config not loaded."""
    cfg = AppConfig()
    cfg._configuration = None
    with pytest.raises(LogicError):
        _ = cfg.rag_id_mapping


def test_score_multiplier_mapping_empty_when_no_byok(minimal_config: AppConfig) -> None:
    """Test that score_multiplier_mapping returns empty dict when no BYOK RAG configured."""
    assert minimal_config.score_multiplier_mapping == {}


def test_score_multiplier_mapping_with_byok_defaults(tmp_path: Path) -> None:
    """Test that score_multiplier_mapping uses default multiplier when not specified."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "byok_rag": [
                {
                    "rag_id": "my-kb",
                    "vector_db_id": "vs-001",
                    "db_path": str(db_file),
                },
            ],
        }
    )
    assert cfg.score_multiplier_mapping == {"vs-001": 1.0}


def test_score_multiplier_mapping_with_custom_values(tmp_path: Path) -> None:
    """Test that score_multiplier_mapping builds correct mapping with custom values."""
    db_file1 = tmp_path / "test1.db"
    db_file1.touch()
    db_file2 = tmp_path / "test2.db"
    db_file2.touch()
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "llama_stack": {
                "api_key": "k",
                "url": "http://test.com:1234",
                "use_as_library_client": False,
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "byok_rag": [
                {
                    "rag_id": "kb1",
                    "vector_db_id": "vs-001",
                    "db_path": str(db_file1),
                    "score_multiplier": 1.5,
                },
                {
                    "rag_id": "kb2",
                    "vector_db_id": "vs-002",
                    "db_path": str(db_file2),
                    "score_multiplier": 0.75,
                },
            ],
        }
    )
    assert cfg.score_multiplier_mapping == {"vs-001": 1.5, "vs-002": 0.75}


def test_score_multiplier_mapping_not_loaded() -> None:
    """Test that score_multiplier_mapping raises when config not loaded."""
    cfg = AppConfig()
    cfg._configuration = None
    with pytest.raises(LogicError):
        _ = cfg.score_multiplier_mapping


wrong_configurations = [
    {
        "name": "Colin Adams",
        "service": {
            "host": "Serve control majority quite approach step.",
            "port": 378,
            "base_url": "Or blood represent beat.",
            "auth_enabled": False,
            "workers": 896,
            "color_log": True,
            "access_log": True,
            "tls_config": {
                "tls_certificate_path": None,
                "tls_key_path": None,
                "tls_key_password": None,
            },
            "root_path": "Word especially structure.",
            "cors": {
                "allow_origins": [
                    "Last drop work less really sister.",
                    "Body light risk edge.",
                ],
                "allow_credentials": True,
                "allow_methods": ["Great teach very staff."],
                "allow_headers": [
                    "Avoid eye space tree minute.",
                    "Wall they realize. Data teach which seek policy ri",
                    "New week public how. Room of line good fire leave.",
                ],
            },
        },
        "llama_stack": {
            "url": "https://www.west.com/",
            "api_key": "api_key",
            "use_as_library_client": False,
            "library_client_config_path": "Strategy stand return catch range professor.",
            "timeout": 486,
        },
        "user_data_collection": {
            "feedback_enabled": True,
            "feedback_storage": None,
            "transcripts_enabled": True,
            "transcripts_storage": "Must risk kid just.",
        },
        "database": {
            "sqlite": {"db_path": "Shake stuff particularly move. Military center sug"},
            "postgres": {
                "host": "Rock either measure leg carry.",
                "port": 91,
                "db": "Institution.",
                "user": "Action production item hour option reflect.",
                "password": "3+37w9K!#3Bk",
                "namespace": None,
                "ssl_mode": "Seek very sell whom. Order dog ready away.",
                "gss_encmode": "A pressure leave but past drive some.",
                "ca_cert_path": None,
            },
        },
        "mcp_servers": [
            {
                "name": "Sherry Greene",
                "provider_id": "Stage tell north despite tell. Such institution ra",
                "url": "https://wilkins.net/",
                "authorization_headers": {
                    "their": "Size right huge both wall financial."
                },
                "headers": [
                    "Throughout speak next.",
                    "Least may discuss name. Whatever bad take.",
                ],
                "timeout": "10",
            },
            {
                "name": "Christopher Cain",
                "provider_id": "Hair raise risk career traditional.",
                "url": "https://cruz.com/",
                "authorization_headers": {
                    "surface": "Whose over special suddenly why. Candidate nearly ",
                    "general": "Others effort analysis significant car maintain.",
                    "return": "Wide enter ago name vote.",
                },
                "headers": ["Him keep finally."],
                "timeout": "10",
            },
            {
                "name": "Eric Martin",
                "provider_id": "Food assume per stop fear lay.",
                "url": "https://solis-porter.com/",
                "authorization_headers": {
                    "us": "Wall building officer father success.",
                    "PM": "Medical perhaps him impact affect.",
                },
                "headers": [
                    "Test owner too side.",
                    "Girl year process team.",
                    "Able computer anyone keep must back finish century",
                ],
                "timeout": "10",
            },
        ],
        "authentication": {
            "module": "Fast learn describe.",
            "skip_tls_verification": True,
            "skip_for_health_probes": False,
            "skip_for_metrics": True,
            "k8s_cluster_api": "http://1.2.3.4",
            "k8s_ca_cert_path": None,
            "jwk_config": None,
            "api_key_config": {"api_key": "key"},
            "rh_identity_config": None,
        },
        "authorization": None,
        "customization": None,
        "inference": {
            "default_model": "Exist important land left peace.",
            "default_provider": "Much us fight suggest.",
        },
        "conversation_cache": {
            "type": None,
            "memory": {"max_entries": 447},
            "sqlite": None,
            "postgres": {
                "host": "View price final number individual. Under directio",
                "port": 343,
                "db": "Quickly lay health stock whose gas born.",
                "user": "From bill attack none.",
                "password": "#4$uLlS2CNSP",
                "namespace": "Task to including nice author.",
                "ssl_mode": "His yard question issue attorney.",
                "gss_encmode": "Decision ability else base pay.",
                "ca_cert_path": "file",
            },
        },
        "byok_rag": [
            {
                "rag_id": "Weight message strong wind land bar.",
                "rag_type": "Learn person tell increase dog even.",
                "embedding_model": "By our television. Southern full a course.",
                "embedding_dimension": 753,
                "vector_db_id": "Indicate see door specific hard region one.",
                "db_path": "A none owner visit wish medical cut Mrs. Later nig",
                "score_multiplier": 388.45,
            }
        ],
        "a2a_state": {"sqlite": None, "postgres": None},
        "quota_handlers": {
            "sqlite": {"db_path": "Experience five able citizen work member call cond"},
            "postgres": {
                "host": "Name garden ready finally century.",
                "port": 722,
                "db": "Fire into list can or.",
                "user": "Collection scientist evening consumer suddenly fac",
                "password": "&#drK6YgvGNh",
                "namespace": "Technology chance quality parent.",
                "ssl_mode": "Simple safe ten general along pull.",
                "gss_encmode": "Listen population.",
                "ca_cert_path": None,
            },
            "limiters": [
                {
                    "type": "user_limiter",
                    "name": "Katelyn Everett",
                    "initial_quota": 651,
                    "quota_increase": 427,
                    "period": "Coach develop ever happen.",
                },
                {
                    "type": "user_limiter",
                    "name": "Alec Sullivan",
                    "initial_quota": 819,
                    "quota_increase": 13,
                    "period": "Enough table best. Work final imagine learn tax su",
                },
            ],
            "scheduler": {
                "period": 415,
                "database_reconnection_count": 545,
                "database_reconnection_delay": 150,
            },
            "enable_token_history": True,
        },
        "azure_entra_id": {
            "tenant_id": "tenant_id",
            "client_id": "client_id",
            "client_secret": "client_secret",
            "scope": "Because its own above.",
        },
        "rlsapi_v1": {
            "allow_verbose_infer": False,
            "quota_subject": "user_id",
        },
        "splunk": {
            "enabled": False,
            "url": "https://cordova-moss.net/",
            "token_path": None,
            "index": "Company but couple or.",
            "source": "Page cell data mission you player. Leg development",
            "timeout": 187,
            "verify_ssl": False,
        },
        "deployment_environment": "Second say body know music while.",
        "rag": {
            "inline": [
                "Local authority pressure pretty. Travel something ",
                "Watch meet able such.",
                "Different apply size.",
            ],
            "tool": [
                "Full develop under his.",
                "Black political father project become.",
                "Once however son place.",
            ],
        },
        "okp": {
            "rhokp_url": None,
            "offline": True,
            "chunk_filter_query": "Foreign space system.",
        },
    },
    {
        "name": "Nathaniel Williams",
        "service": {
            "host": "Response although crime less.",
            "port": 275,
            "base_url": None,
            "auth_enabled": False,
            "workers": 218,
            "color_log": False,
            "access_log": True,
            "tls_config": {
                "tls_certificate_path": None,
                "tls_key_path": None,
                "tls_key_password": "password",
            },
            "root_path": "Model something information strong focus.",
            "cors": {
                "allow_origins": [
                    "Clear yes hand language.",
                    "Expect will research bag. Standard exist property ",
                    "Cultural their pass explain.",
                ],
                "allow_credentials": False,
                "allow_methods": [
                    "Wind hit work arm man.",
                    "Cost food serve national education. Painting wide ",
                ],
                "allow_headers": ["At professor seek hospital eat."],
            },
        },
        "llama_stack": {
            "url": None,
            "api_key": None,
            "use_as_library_client": None,
            "library_client_config_path": None,
            "timeout": 890,
        },
        "user_data_collection": {
            "feedback_enabled": True,
            "feedback_storage": "Travel consider themselves trouble there budget.",
            "transcripts_enabled": True,
            "transcripts_storage": None,
        },
        "database": {
            "sqlite": {"db_path": "Will yet other need."},
            "postgres": {
                "host": "Bar various source each.",
                "port": 236,
                "db": "Positive federal organization between knowledge so",
                "user": "Score girl far party example care leave.",
                "password": "mOW_ES5n7TB6",
                "namespace": "If act test financial.",
                "ssl_mode": "Whether bad still use.",
                "gss_encmode": "Easy agreement authority again rest.",
                "ca_cert_path": None,
            },
        },
        "mcp_servers": [
            {
                "name": "William Roberts",
                "provider_id": "Represent surface stuff.",
                "url": "https://www.rodriguez-cross.com/",
                "authorization_headers": {
                    "a": "Technology sure event third go for institution mon",
                    "yourself": "Physical will coach.",
                    "size": "Start wife body.",
                },
                "headers": [
                    "Be recognize civil.",
                    "Senior system give.",
                    "Son often them resource her star thus.",
                ],
                "timeout": None,
            },
            {
                "name": "William Maldonado",
                "provider_id": "Yet our town hope.",
                "url": "https://collins.com/",
                "authorization_headers": {
                    "then": "Water we provide those leader less chance.",
                    "whom": "Theory single common sing what for.",
                    "manage": "Keep thousand ground.",
                },
                "headers": [
                    "Those manage area light. Large authority lawyer le",
                    "Bed speech adult imagine office.",
                    "Decide analysis.",
                ],
                "timeout": "-10",
            },
            {
                "name": "Ashley Walker",
                "provider_id": "Case film eight early.",
                "url": "https://www.davis-wilson.net/",
                "authorization_headers": {
                    "amount": "Customer religious let ever.",
                    "deep": "Nothing indeed but argue single.",
                },
                "headers": ["Call street easy lawyer quite enjoy once task. Mil"],
                "timeout": "-10",
            },
        ],
        "authentication": {
            "module": "Arrive blue lose rock.",
            "skip_tls_verification": False,
            "skip_for_health_probes": True,
            "skip_for_metrics": False,
            "k8s_cluster_api": None,
            "k8s_ca_cert_path": None,
            "jwk_config": {
                "url": "https://davis.org/",
                "jwt_configuration": {
                    "user_id_claim": "Reason million financial raise environmental.",
                    "username_claim": "Wonder activity yes.",
                    "role_rules": [
                        {
                            "jsonpath": "Social bank buy.",
                            "operator": "-10",
                            "negate": True,
                            "value": "any_value_placeholder",
                            "roles": ["Part house prevent story."],
                        },
                        {
                            "jsonpath": "Outside how note when. Mr entire affect usually.",
                            "operator": "-10",
                            "negate": False,
                            "value": "any_value_placeholder",
                            "roles": [
                                "Store child serious.",
                                "Song plant sense technology though consider. Paren",
                            ],
                        },
                        {
                            "jsonpath": "Family light power color number. Treat court stuff",
                            "operator": "-10",
                            "negate": False,
                            "value": "any_value_placeholder",
                            "roles": [
                                "Television set over offer task north card those."
                            ],
                        },
                    ],
                },
            },
            "api_key_config": None,
            "rh_identity_config": None,
        },
        "authorization": {
            "access_rules": [
                {"role": "Perhaps idea nothing attorney word.", "actions": ["action1"]},
                {
                    "role": "Somebody operation involve alone. Born example sea",
                    "actions": ["action1", "action2"],
                },
                {
                    "role": "Watch most resource word information allow ability",
                    "actions": ["action1"],
                },
            ]
        },
        "customization": {
            "profile_path": None,
            "disable_query_system_prompt": False,
            "disable_shield_ids_override": True,
            "system_prompt_path": "/",
            "system_prompt": None,
            "agent_card_path": "/",
            "agent_card_config": None,
            "custom_profile": "/",
        },
        "inference": {
            "default_model": "Fill popular near avoid family year.",
            "default_provider": "Pressure former operation table start president.",
        },
        "conversation_cache": {
            "type": "noop",
            "memory": None,
            "sqlite": {"db_path": "Own relationship back production."},
            "postgres": {
                "host": "Religious author little. Thought bring agency rais",
                "port": 146,
                "db": "Other safe resource stop.",
                "user": "Concern center once music activity bad thousand.",
                "password": "zHDCzco5^S2Z",
                "namespace": None,
                "ssl_mode": "Run claim process large.",
                "gss_encmode": "Tv east art pattern.",
                "ca_cert_path": "certs",
            },
        },
        "byok_rag": [
            {
                "rag_id": "Tonight relate there record.",
                "rag_type": "Politics development real play main chair capital ",
                "embedding_model": "Prepare memory outside.",
                "embedding_dimension": 449,
                "vector_db_id": "Political right gun law public group rock.",
                "db_path": "Consider still recognize church. Area suggest noth",
                "score_multiplier": 183.85,
            },
            {
                "rag_id": "One again under respond poor beyond.",
                "rag_type": "Six base physical.",
                "embedding_model": "Surface that choice.",
                "embedding_dimension": 736,
                "vector_db_id": "Forget level other agreement.",
                "db_path": "Argue pull out race town.",
                "score_multiplier": 225.21,
            },
        ],
        "a2a_state": {"sqlite": None, "postgres": None},
        "quota_handlers": {
            "sqlite": None,
            "postgres": {
                "host": "List fund yes kitchen meet southern.",
                "port": 337,
                "db": "Five rest behavior tonight couple.",
                "user": "Message accept whom authority.",
                "password": "B28qYhFt$nnI",
                "namespace": None,
                "ssl_mode": "Teach wide worry tend.",
                "gss_encmode": "Example weight window.",
                "ca_cert_path": "/",
            },
            "limiters": [
                {
                    "type": "user_limiter",
                    "name": "Michael Jackson",
                    "initial_quota": 371,
                    "quota_increase": 210,
                    "period": "Start skill miss economy know.",
                },
                {
                    "type": "user_limiter",
                    "name": "Richard Moore",
                    "initial_quota": 189,
                    "quota_increase": 83,
                    "period": "Another maybe certainly week trouble.",
                },
                {
                    "type": "user_limiter",
                    "name": "Charlene Perez",
                    "initial_quota": 387,
                    "quota_increase": 8,
                    "period": "Music difficult enough.",
                },
            ],
            "scheduler": {
                "period": 825,
                "database_reconnection_count": 896,
                "database_reconnection_delay": 669,
            },
            "enable_token_history": False,
        },
        "azure_entra_id": {
            "tenant_id": "id",
            "client_id": "id",
            "client_secret": "secrer",
            "scope": "Claim thought seat use.",
        },
        "rlsapi_v1": {"allow_verbose_infer": False, "quota_subject": None},
        "splunk": {
            "enabled": True,
            "url": "https://george-ferguson.org/",
            "token_path": "/",
            "index": None,
            "source": "Successful impact understand least generation cust",
            "timeout": 850,
            "verify_ssl": False,
        },
        "deployment_environment": "Vote mean answer simply turn project.",
        "rag": {
            "inline": [
                "Billion job provide take other.",
                "Eight total figure surface development include out",
                "Which from cover not choice bring sister front.",
            ],
            "tool": ["Ground appear group institution."],
        },
        "okp": {"rhokp_url": None, "offline": False, "chunk_filter_query": None},
    },
    {
        "name": "Patricia Henderson",
        "service": {
            "host": "High politics role market party factor yourself. H",
            "port": 65,
            "base_url": "Song list by notice decide politics.",
            "auth_enabled": True,
            "workers": 248,
            "color_log": True,
            "access_log": False,
            "tls_config": {
                "tls_certificate_path": None,
                "tls_key_path": None,
                "tls_key_password": "xyzzy",
            },
            "root_path": "Another budget miss with my pretty.",
            "cors": {
                "allow_origins": ["Edge yes right already."],
                "allow_credentials": True,
                "allow_methods": ["Include game information. Become mother today."],
                "allow_headers": [
                    "Recent crime seem ten.",
                    "Yeah fear design few.",
                    "Information story.",
                ],
            },
        },
        "llama_stack": {
            "url": "http://www.cameron.com/",
            "api_key": "xyzzy",
            "use_as_library_client": False,
            "library_client_config_path": "Write past admit hand area surface.",
            "timeout": 237,
        },
        "user_data_collection": {
            "feedback_enabled": True,
            "feedback_storage": "Poor have happen.",
            "transcripts_enabled": True,
            "transcripts_storage": "Ask sport choice along. Ever concern table traditi",
        },
        "database": {"sqlite": None, "postgres": None},
        "mcp_servers": [
            {
                "name": "Colleen Villanueva",
                "provider_id": "Every present recently.",
                "url": "https://www.taylor.com/",
                "authorization_headers": {
                    "four": "Have camera third when across too.",
                    "dinner": "Field run hair.",
                    "machine": "Time find prove war.",
                },
                "headers": [
                    "Author heavy now start allow focus.",
                    "Eat before by.",
                    "Serious best talk never.",
                ],
                "timeout": None,
            }
        ],
        "authentication": {
            "module": "Offer to trouble off chance personal.",
            "skip_tls_verification": False,
            "skip_for_health_probes": False,
            "skip_for_metrics": True,
            "k8s_cluster_api": None,
            "k8s_ca_cert_path": "xyzzy",
            "jwk_config": None,
            "api_key_config": None,
            "rh_identity_config": None,
        },
        "authorization": {
            "access_rules": [
                {
                    "role": "Nothing development house computer.",
                    "actions": ["xyzzy", "xyzzy", "xyzzy"],
                },
                {
                    "role": "Sea coach without wide audience pretty fine.",
                    "actions": ["xyzzy", "xyzzy"],
                },
                {
                    "role": "Will Congress style picture deal. Explain set assu",
                    "actions": ["xyzzy"],
                },
            ]
        },
        "customization": {
            "profile_path": None,
            "disable_query_system_prompt": False,
            "disable_shield_ids_override": False,
            "system_prompt_path": "xyzzy",
            "system_prompt": None,
            "agent_card_path": "xyzzy",
            "agent_card_config": None,
            "custom_profile": None,
        },
        "inference": {
            "default_model": "Television someone about wall join.",
            "default_provider": None,
        },
        "conversation_cache": {
            "type": None,
            "memory": None,
            "sqlite": None,
            "postgres": None,
        },
        "byok_rag": [
            {
                "rag_id": "Something worker campaign war through.",
                "rag_type": "Check simple since next then statement.",
                "embedding_model": "Class third author series.",
                "embedding_dimension": 211,
                "vector_db_id": "Less put site alone amount.",
                "db_path": "Live child most throughout.",
                "score_multiplier": 252.41,
            }
        ],
        "a2a_state": {"sqlite": None, "postgres": None},
        "quota_handlers": {
            "sqlite": None,
            "postgres": None,
            "limiters": [
                {
                    "type": "xyzzy",
                    "name": "Jerry Brown",
                    "initial_quota": 532,
                    "quota_increase": 509,
                    "period": "After determine almost make yeah support. Away tak",
                },
                {
                    "type": "xyzzy",
                    "name": "James Martin",
                    "initial_quota": 167,
                    "quota_increase": 278,
                    "period": "Book gas exist these.",
                },
                {
                    "type": "xyzzy",
                    "name": "Mr. Douglas Kelly DDS",
                    "initial_quota": 210,
                    "quota_increase": 930,
                    "period": "Section many southern new.",
                },
            ],
            "scheduler": {
                "period": 984,
                "database_reconnection_count": 118,
                "database_reconnection_delay": 961,
            },
            "enable_token_history": False,
        },
        "azure_entra_id": {
            "tenant_id": "xyzzy",
            "client_id": "xyzzy",
            "client_secret": "xyzzy",
            "scope": "Spring billion represent town actually serious mor",
        },
        "rlsapi_v1": {"allow_verbose_infer": False, "quota_subject": "xyzzy"},
        "splunk": {
            "enabled": False,
            "url": "http://www.bell.com/",
            "token_path": "xyzzy",
            "index": None,
            "source": "Risk condition boy conference particularly control",
            "timeout": 398,
            "verify_ssl": False,
        },
        "deployment_environment": "Mouth view form.",
        "rag": {
            "inline": [
                "Interesting during product himself attack Democrat",
                "Decision I order particularly.",
                "Couple reflect relate two agree local.",
            ],
            "tool": ["Her society move lay.", "Network material like."],
        },
        "okp": {
            "rhokp_url": "xyzzy",
            "offline": False,
            "chunk_filter_query": "Beautiful society within.",
        },
    },
]


@pytest.mark.parametrize("config_dict", wrong_configurations)
def test_init_from_dict_fake_data(config_dict: dict[str, Any]) -> None:
    """Test the configuration initialization from dictionary with config values."""
    with pytest.raises(ValueError):
        cfg = AppConfig()
        cfg.init_from_dict(config_dict)
