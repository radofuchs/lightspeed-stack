@e2e_group_2 @Authorized
Feature: Prompts API endpoint tests

  Background:
    Given The service is started locally
      And The system is in default state
      And I set the Authorization header to Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ikpva
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"
      And The service uses the lightspeed-stack-auth-noop-token.yaml configuration
      And The service is restarted

  Scenario: Create prompt returns 200 with prompt_id and version 1
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    And The prompt version in the response is 1

  Scenario: List prompts includes a newly created prompt
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    When I access REST API endpoint "prompts" using HTTP GET method
    Then The status code of the response is 200
    And The prompts list contains the stored prompt id

  Scenario: Get prompt by id returns latest version
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    When I access REST API prompts endpoint with stored prompt id using HTTP GET method
    Then The status code of the response is 200
    And The prompt_id in the response matches the stored prompt id
    And The prompt version in the response is 1

  Scenario: Update prompt increments version
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    When I access REST API prompts endpoint with stored prompt id using HTTP PUT method
    """
    {"prompt": "Summarize in bullets: {{text}}", "version": 1, "set_as_default": true, "variables": ["text"]}
    """
    Then The status code of the response is 200
    And The prompt_id in the response matches the stored prompt id
    And The prompt version in the response is 2

  Scenario: Get prompt by id with version query returns that version
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    When I access REST API prompts endpoint with stored prompt id using HTTP PUT method
    """
    {"prompt": "Summarize in bullets: {{text}}", "version": 1, "set_as_default": true, "variables": ["text"]}
    """
    Then The status code of the response is 200
    And The prompt version in the response is 2
    When I access REST API prompts endpoint with stored prompt id and version 1 using HTTP GET method
    Then The status code of the response is 200
    And The prompt_id in the response matches the stored prompt id
    And The prompt version in the response is 1

  Scenario: Delete prompt returns deleted then not found on second delete
    When I access REST API endpoint "prompts" using HTTP POST method
    """
    {"prompt": "Summarize: {{text}}", "variables": ["text"]}
    """
    Then The status code of the response is 200
    And I store the prompt_id from the last response
    When I access REST API prompts endpoint with stored prompt id using HTTP DELETE method
    Then The status code of the response is 200
    And The prompt_id in the response matches the stored prompt id
    And The body of the response contains deleted successfully
    When I access REST API prompts endpoint with stored prompt id using HTTP DELETE method
    Then The status code of the response is 200
    And The prompt_id in the response matches the stored prompt id
    And The body of the response contains Prompt not found

  Scenario: Invalid prompt_id returns bad request
    When I access REST API endpoint "prompts/not_a_prompt_id" using HTTP GET method
    Then The status code of the response is 400
    And The body of the response contains Invalid prompt ID format
