"""Unit tests for skills endpoint."""

from pathlib import Path

import pytest
from fastapi import Request
from pytest_mock import MockerFixture

from app.endpoints.skills import skills_endpoint_handler
from authentication.interface import AuthTuple
from models.api.responses.successful import SkillsResponse
from models.config import SkillsConfiguration
from tests.unit.utils.auth_helpers import mock_authorization_resolvers

MOCK_AUTH: AuthTuple = ("mock_user_id", "mock_username", True, "mock_token")


@pytest.mark.asyncio
async def test_skills_loaded(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Test that loaded skills are returned with name and description."""
    mock_authorization_resolvers(mocker)

    skills_root = tmp_path / "skills"
    for name, desc in [
        ("code-review", "Review code for quality and security"),
        ("openshift-troubleshooting", "Troubleshoot OpenShift cluster issues"),
    ]:
        skill_dir = skills_root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\nInstructions.\n",
            encoding="utf-8",
        )

    skills_config = SkillsConfiguration(paths=[skills_root])
    mock_config = mocker.patch("app.endpoints.skills.configuration")
    mock_config.configuration.skills = skills_config

    request = Request(scope={"type": "http"})
    response = await skills_endpoint_handler(auth=MOCK_AUTH, request=request)

    assert isinstance(response, SkillsResponse)
    assert len(response.skills) == 2
    names = {s["name"] for s in response.skills}
    assert names == {"code-review", "openshift-troubleshooting"}
    for skill in response.skills:
        assert "name" in skill
        assert "description" in skill


@pytest.mark.asyncio
async def test_no_skills_configured(
    mocker: MockerFixture,
) -> None:
    """Test that an empty list is returned when no skills are configured."""
    mock_authorization_resolvers(mocker)

    mock_config = mocker.patch("app.endpoints.skills.configuration")
    mock_config.configuration.skills = None

    request = Request(scope={"type": "http"})
    response = await skills_endpoint_handler(auth=MOCK_AUTH, request=request)

    assert isinstance(response, SkillsResponse)
    assert response.skills == []


@pytest.mark.asyncio
async def test_empty_skills_paths(
    mocker: MockerFixture,
) -> None:
    """Test that an empty list is returned when skills paths are empty."""
    mock_authorization_resolvers(mocker)

    mock_config = mocker.patch("app.endpoints.skills.configuration")
    mock_config.configuration.skills = SkillsConfiguration(paths=[])

    request = Request(scope={"type": "http"})
    response = await skills_endpoint_handler(auth=MOCK_AUTH, request=request)

    assert isinstance(response, SkillsResponse)
    assert response.skills == []


@pytest.mark.asyncio
async def test_skills_with_references(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Test that skills with references/ subdirectory are listed correctly."""
    mock_authorization_resolvers(mocker)

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "rhdh-dynamic-plugins"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rhdh-dynamic-plugins\ndescription: RHDH dynamic plugins guide\n---\n\nInstructions.\n",
        encoding="utf-8",
    )
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "plugin-list.md").write_text("# Plugins\n- plugin-a\n", encoding="utf-8")

    skills_config = SkillsConfiguration(paths=[skills_root])
    mock_config = mocker.patch("app.endpoints.skills.configuration")
    mock_config.configuration.skills = skills_config

    request = Request(scope={"type": "http"})
    response = await skills_endpoint_handler(auth=MOCK_AUTH, request=request)

    assert isinstance(response, SkillsResponse)
    assert len(response.skills) == 1
    assert response.skills[0]["name"] == "rhdh-dynamic-plugins"
    assert response.skills[0]["description"] == "RHDH dynamic plugins guide"
