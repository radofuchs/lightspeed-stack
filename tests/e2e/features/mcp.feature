@e2e_group_2
Feature: MCP tests

  Background:
    Given The service is started locally
      And The system is in default state
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"


# File-based (valid token) — lightspeed-stack-mcp-file-auth.yaml
  @MCPFileAuthConfig
  Scenario: Check if tools endpoint succeeds when MCP file-based auth token is passed
    Given MCP toolgroups are reset for a new MCP configuration
      And The service uses the lightspeed-stack-mcp-file-auth.yaml configuration
      And The service is restarted
    And The mcp-file mcp server Authorization header is set to "/tmp/mcp-token"
    When I access REST API endpoint "tools" using HTTP GET method
    Then The status code of the response is 200
    And The body of the response contains mcp-file