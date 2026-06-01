"""Unit tests for QuotaHandlersConfiguration model."""

# pylint: disable=no-member
import pytest
from pydantic import ValidationError
from pytest_subtests import SubTests

from models.config import QuotaHandlersConfiguration, QuotaSchedulerConfiguration


def test_quota_handlers_configuration(subtests: SubTests) -> None:
    """Test the quota handlers configuration."""
    with subtests.test(msg="Token history disabled"):
        cfg = QuotaHandlersConfiguration(
            sqlite=None,
            postgres=None,
            limiters=[],
            scheduler=QuotaSchedulerConfiguration(
                database_reconnection_count=10,
                database_reconnection_delay=60,
                period=10,
            ),
            enable_token_history=False,
        )
        assert cfg is not None
        assert cfg.sqlite is None
        assert cfg.postgres is None
        assert cfg.limiters == []
        assert cfg.scheduler is not None
        assert cfg.scheduler.database_reconnection_count == 10
        assert cfg.scheduler.database_reconnection_delay == 60
        assert cfg.scheduler.period == 10
        assert not cfg.enable_token_history

    with subtests.test(msg="Token history enabled"):
        cfg = QuotaHandlersConfiguration(
            sqlite=None,
            postgres=None,
            limiters=[],
            scheduler=QuotaSchedulerConfiguration(
                database_reconnection_count=10,
                database_reconnection_delay=60,
                period=10,
            ),
            enable_token_history=True,
        )
        assert cfg is not None
        assert cfg.sqlite is None
        assert cfg.postgres is None
        assert cfg.limiters == []
        assert cfg.scheduler is not None
        assert cfg.scheduler.database_reconnection_count == 10
        assert cfg.scheduler.database_reconnection_delay == 60
        assert cfg.scheduler.period == 10
        assert cfg.enable_token_history


def test_quota_handlers_scheduler_configuration(subtests: SubTests) -> None:
    """Test the quota handlers configuration."""
    with subtests.test(msg="Different reconnection_count"):
        cfg = QuotaHandlersConfiguration(
            sqlite=None,
            postgres=None,
            limiters=[],
            scheduler=QuotaSchedulerConfiguration(
                database_reconnection_count=42,
                database_reconnection_delay=60,
                period=10,
            ),
            enable_token_history=True,
        )
        assert cfg is not None
        assert cfg.sqlite is None
        assert cfg.postgres is None
        assert cfg.limiters == []
        assert cfg.scheduler is not None
        assert cfg.scheduler.database_reconnection_count == 42
        assert cfg.scheduler.database_reconnection_delay == 60
        assert cfg.scheduler.period == 10
        assert cfg.enable_token_history

    with subtests.test(msg="Different reconnection_delay"):
        cfg = QuotaHandlersConfiguration(
            sqlite=None,
            postgres=None,
            limiters=[],
            scheduler=QuotaSchedulerConfiguration(
                database_reconnection_count=10,
                database_reconnection_delay=42,
                period=10,
            ),
            enable_token_history=True,
        )
        assert cfg is not None
        assert cfg.sqlite is None
        assert cfg.postgres is None
        assert cfg.limiters == []
        assert cfg.scheduler is not None
        assert cfg.scheduler.database_reconnection_count == 10
        assert cfg.scheduler.database_reconnection_delay == 42
        assert cfg.scheduler.period == 10
        assert cfg.enable_token_history

    with subtests.test(msg="Different period"):
        cfg = QuotaHandlersConfiguration(
            sqlite=None,
            postgres=None,
            limiters=[],
            scheduler=QuotaSchedulerConfiguration(
                database_reconnection_count=10,
                database_reconnection_delay=60,
                period=600,
            ),
            enable_token_history=True,
        )
        assert cfg is not None
        assert cfg.sqlite is None
        assert cfg.postgres is None
        assert cfg.limiters == []
        assert cfg.scheduler is not None
        assert cfg.scheduler.database_reconnection_count == 10
        assert cfg.scheduler.database_reconnection_delay == 60
        assert cfg.scheduler.period == 600
        assert cfg.enable_token_history


def test_quota_handlers_configuration_improper_values(subtests: SubTests) -> None:
    """Test the quota handlers configuration."""
    with subtests.test(msg="Improper SQLite settings"):
        with pytest.raises(
            ValidationError,
            match="Input should be a valid dictionary or instance of SQLiteDatabaseConfiguration",
        ):
            QuotaHandlersConfiguration(
                sqlite="not a proper config",  # pyright: ignore[reportArgumentType]
                postgres=None,
                limiters=[],
                scheduler=QuotaSchedulerConfiguration(
                    database_reconnection_count=10,
                    database_reconnection_delay=60,
                    period=10,
                ),
                enable_token_history=False,
            )

    with subtests.test(msg="Improper PostgreSQL settings"):
        with pytest.raises(
            ValidationError,
            match="should be a valid dictionary or instance of PostgreSQLDatabaseConfiguration",
        ):
            QuotaHandlersConfiguration(
                sqlite=None,
                postgres="not a proper config",  # pyright: ignore[reportArgumentType]
                limiters=[],
                scheduler=QuotaSchedulerConfiguration(
                    database_reconnection_count=10,
                    database_reconnection_delay=60,
                    period=10,
                ),
                enable_token_history=False,
            )

    with subtests.test(msg="None limiters settings"):
        with pytest.raises(
            ValidationError,
            match="Input should be a valid list",
        ):
            QuotaHandlersConfiguration(
                sqlite=None,
                postgres=None,
                limiters=None,  # pyright: ignore[reportArgumentType]
                scheduler=QuotaSchedulerConfiguration(
                    database_reconnection_count=10,
                    database_reconnection_delay=60,
                    period=10,
                ),
                enable_token_history=False,
            )

    with subtests.test(msg="Improper limiters settings"):
        with pytest.raises(
            ValidationError,
            match="Input should be a valid list",
        ):
            QuotaHandlersConfiguration(
                sqlite=None,
                postgres=None,
                limiters="should be a list of limiters",  # pyright: ignore[reportArgumentType]
                scheduler=QuotaSchedulerConfiguration(
                    database_reconnection_count=10,
                    database_reconnection_delay=60,
                    period=10,
                ),
                enable_token_history=False,
            )

    with subtests.test(msg="Improper enable_token_history"):
        with pytest.raises(
            ValidationError,
            match="Input should be a valid boolean",
        ):
            QuotaHandlersConfiguration(
                sqlite=None,
                postgres=None,
                limiters=[],
                scheduler=QuotaSchedulerConfiguration(
                    database_reconnection_count=10,
                    database_reconnection_delay=60,
                    period=10,
                ),
                enable_token_history="enabled",  # pyright: ignore[reportArgumentType]
            )

    with subtests.test(msg="Improper scheduler"):
        with pytest.raises(
            ValidationError,
            match="Input should be a valid dictionary or instance of QuotaSchedulerConfiguration",
        ):
            QuotaHandlersConfiguration(
                sqlite=None,
                postgres=None,
                limiters=[],
                scheduler="scheduler",  # pyright: ignore[reportArgumentType]
                enable_token_history=False,
            )
