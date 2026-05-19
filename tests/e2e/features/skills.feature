@e2e_group_2 @skip
Feature: Agent skills tests

  Background:
    Given The service is started locally
      And The system is in default state
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"

  # --- Skill tools registration ---

  @SkillsConfig
  Scenario: Skill tools are registered when skills are configured
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills.yaml configuration
      And The service is restarted
    When I access REST API endpoint "tools" using HTTP GET method
    Then The status code of the response is 200
     And The body of the response is the following    #TODO: Currently placeholder, should reflect actual tools (all tools not just skill tools)
      """
      {
        "tools": [
            {
                "identifier": "filesystem_read",
                "description": "Read contents of a file from the filesystem",
                "parameters": [
                    {
                        "name": "path",
                        "description": "Path to the file to read",
                        "parameter_type": "string",
                        "required": True,
                        "default": None,
                    }
                ],
                "provider_id": "model-context-protocol",
                "toolgroup_id": "filesystem-tools",
                "server_source": "http://localhost:3000",
                "type": "tool",
            }
        ],
      }
      """

  Scenario: Skill tools are not registered when no skills are configured
    Given The service uses the lightspeed-stack.yaml configuration
      And The service is restarted
    When I access REST API endpoint "tools" using HTTP GET method
    Then The status code of the response is 200
     And The body of the response is the following    #TODO: Currently placeholder, should reflect actual tools (default tools, not skill tools)
      """
      {
        "tools": [
            {
                "identifier": "filesystem_read",
                "description": "Read contents of a file from the filesystem",
                "parameters": [
                    {
                        "name": "path",
                        "description": "Path to the file to read",
                        "parameter_type": "string",
                        "required": True,
                        "default": None,
                    }
                ],
                "provider_id": "model-context-protocol",
                "toolgroup_id": "filesystem-tools",
                "server_source": "http://localhost:3000",
                "type": "tool",
            }
        ],
      }
      """

  # --- Skill discovery ---

  @SkillsConfig
  Scenario: LLM can discover skills via list_skills tool using query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "query" to ask question 
    """
    {"query": "What skills are available? Use the list_skills tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased

  @SkillsConfig
  Scenario: LLM can discover skills via list_skills tool using streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "streaming_query" to ask question 
    """
    {"query": "What skills are available? Use the list_skills tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased

  # --- Skill activation ---

  @SkillsConfig
  Scenario: LLM can activate a skill and use its instructions via query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "query" to ask question 
    """
    {"query": "I need help with e2e testing. Use the activate_skill tool to load the e2e-test-skill.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased

  @SkillsConfig
  Scenario: LLM can activate a skill and use its instructions via streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "streaming_query" to ask question 
    """
    {"query": "I need help with e2e testing. Use the activate_skill tool to load the e2e-test-skill.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased


  # --- Skill resource loading ---

  @SkillsConfig
  Scenario: LLM can load a skill reference file via load_skill_resource tool using query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "query" to ask question 
    """
    {"query": "Load the reference file references/guide.md from the e2e-test-skill using load_skill_resource.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased

  @SkillsConfig
  Scenario: LLM can load a skill reference file via load_skill_resource tool using streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "streaming_query" to ask question 
    """
    {"query": "Load the reference file references/guide.md from the e2e-test-skill using load_skill_resource.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
      And The token metrics have increased

  # --- Error handling: unknown skill ---

  @SkillsConfig
  Scenario: activate_skill returns error for unknown skill name via query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
    When I use "query" to ask question 
    """
    {"query": "Activate a skill called nonexistent-skill using the activate_skill tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "failure",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """

  @SkillsConfig
  Scenario: activate_skill returns error for unknown skill name via streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
    When I use "streaming_query" to ask question 
    """
    {"query": "Activate a skill called nonexistent-skill using the activate_skill tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "failure",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
  # --- Error handling: missing resource ---

  @SkillsConfig
  Scenario: load_skill_resource returns error for nonexistent resource file via query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
    When I use "query" to ask question 
    """
    {"query": "Load references/nonexistent.md from e2e-test-skill using load_skill_resource.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "failure",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """

  @SkillsConfig
  Scenario: load_skill_resource returns error for nonexistent resource file via streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
    When I use "streaming_query" to ask question 
    """
    {"query": "Load references/nonexistent.md from e2e-test-skill using load_skill_resource.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "failure",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """


  # --- Context management: deduplication ---

  @SkillsConfig
  Scenario: Duplicate skill activation in same conversation returns already-loaded note via query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
    When I use "query" to ask question 
    """
    {"query": "Activate e2e-test-skill using the activate_skill tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And I store conversation details
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """

    When I use "query" to ask question with same conversation_id
    """
    {"query": "Activate e2e-test-skill again using the activate_skill tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "failure",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """


  # --- Multiple skills ---

  @SkillsMultiConfig
  Scenario: Skills directory path discovers all skills in subdirectories via query endpoint
    Given The e2e-test-skill skill directory path is "skills/e2e-test-skill"
      And The e2e-second-skill skill directory path is "skills/e2e-second-skill"
      And The service uses the lightspeed-stack-skills-directory.yaml configuration
      And The service is restarted
    When I use "query" to ask question 
    """
    {"query": "List all available skills using the list_skills tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """

  @SkillsMultiConfig
  Scenario: Skills directory path discovers all skills in subdirectories via streaming_query endpoint
    Given The e2e-test-skill skill directory path is "skills/e2e-test-skill"
      And The e2e-second-skill skill directory path is "skills/e2e-second-skill"
      And The service uses the lightspeed-stack-skills-directory.yaml configuration
      And The service is restarted
    When I use "streaming_query" to ask question 
    """
    {"query": "List all available skills using the list_skills tool.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The response is the last streamed fragment
      And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "1",
          "status": "success",
          "content": "bla",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
