"""Unit tests for QuotaHandlersConfiguration model."""

# pylint: disable=no-member
from typing import Any

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
