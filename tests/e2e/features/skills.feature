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
            "identifier": "insert_into_memory",
            "description": "Insert documents into memory",
            "parameters": [],
            "provider_id": "rag-runtime",
            "toolgroup_id": "builtin::rag",
            "server_source": "builtin",
            "type": "tool_group"
          },
          {
            "identifier": "knowledge_search",
            "description": "Search for information in a database.",
            "parameters": [
              {
                "name": "query",
                "description": "The query to search for. Can be a natural language sentence or keywords.",
                "parameter_type": "string",
                "required": true,
                "default": null
              }
            ],
            "provider_id": "rag-runtime",
            "toolgroup_id": "builtin::rag",
            "server_source": "builtin",
            "type": "tool_group"
          },
          {
            "identifier": "list_skills",
            "description": "List available skills with their names and descriptions. Call this to discover what skills are available.",
            "parameters": [],
            "provider_id": "agent-skills",
            "toolgroup_id": "builtin::agent-skills",
            "server_source": "builtin",
            "type": "tool"
          },
          {
            "identifier": "activate_skill",
            "description": "Load full instructions for a skill. Call this when a task matches a skill's description.",
            "parameters": [
              {
                "name": "name",
                "description": "The name of the skill to load",
                "parameter_type": "string",
                "required": true,
                "default": null
              }
            ],
            "provider_id": "agent-skills",
            "toolgroup_id": "builtin::agent-skills",
            "server_source": "builtin",
            "type": "tool"
          },
          {
            "identifier": "load_skill_resource",
            "description": "Load a file from a skill's references/ directory. Use this when skill instructions reference additional documentation.",
            "parameters": [
              {
                "name": "skill_name",
                "description": "The name of the skill containing the resource",
                "parameter_type": "string",
                "required": true,
                "default": null
              },
              {
                "name": "path",
                "description": "Relative path to the resource file (e.g., 'references/guide.md')",
                "parameter_type": "string",
                "required": true,
                "default": null
              }
            ],
            "provider_id": "agent-skills",
            "toolgroup_id": "builtin::agent-skills",
            "server_source": "builtin",
            "type": "tool"
          },
          {
            "identifier": "run_skill_script",
            "description": "Execute a skill script that performs actions or computations.",
            "parameters": [
              {
                "name": "skill_name",
                "description": "Name of the skill containing the script",
                "parameter_type": "string",
                "required": true,
                "default": null
              },
              {
                "name": "script_name",
                "description": "Exact name of the script as listed in the skill",
                "parameter_type": "string",
                "required": true,
                "default": null
              },
              {
                "name": "args",
                "description": "Arguments required by the script",
                "parameter_type": "object",
                "required": false,
                "default": null
              }
            ],
            "provider_id": "agent-skills",
            "toolgroup_id": "builtin::agent-skills",
            "server_source": "builtin",
            "type": "tool"
          }
        ]
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
            "identifier": "insert_into_memory",
            "description": "Insert documents into memory",
            "parameters": [],
            "provider_id": "rag-runtime",
            "toolgroup_id": "builtin::rag",
            "server_source": "builtin",
            "type": "tool_group"
          },
          {
            "identifier": "knowledge_search",
            "description": "Search for information in a database.",
            "parameters": [
              {
                "name": "query",
                "description": "The query to search for. Can be a natural language sentence or keywords.",
                "parameter_type": "string",
                "required": true,
                "default": null
              }
            ],
            "provider_id": "rag-runtime",
            "toolgroup_id": "builtin::rag",
            "server_source": "builtin",
            "type": "tool_group"
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
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        }      ]
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
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "failure",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
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
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """

  # --- Full progressive disclosure flow ---

  @SkillsConfig @flaky
  Scenario: LLM completes list_skills then activate_skill then load_skill_resource via query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "query" to ask question
    """
    {"query": "Use the agent skills tools in this exact order: (1) call list_skills to discover available skills, (2) call activate_skill with name \"e2e-test-skill\" to load its instructions, (3) call load_skill_resource with skill_name \"e2e-test-skill\" and path \"references/guide.md\" to read the reference guide. After all three tool calls complete, briefly summarize the guide.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        },
        {
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        },
        {
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """


  @SkillsConfig @flaky
  Scenario: LLM completes list_skills then activate_skill then load_skill_resource via streaming_query endpoint
    Given The e2e-test-skill skill directory path is "e2e-test-skill"
      And The service uses the lightspeed-stack-skills-auth-noop-token.yaml configuration
      And The service is restarted
      And I capture the current token metrics
    When I use "streaming_query" to ask question
    """
    {"query": "Use the agent skills tools in this exact order: (1) call list_skills to discover available skills, (2) call activate_skill with name \"e2e-test-skill\" to load its instructions, (3) call load_skill_resource with skill_name \"e2e-test-skill\" and path \"references/guide.md\" to read the reference guide. After all three tool calls complete, briefly summarize the guide.", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
     And The response is the last streamed fragment
     And The body of the "tool_results" field is    #TODO: Currently placeholder, should reflect actual tool results
      """
      [
        {
          "id": "<call_id>",
          "name": "list_skills"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        },
        {
          "id": "<call_id>",
          "name": "activate_skill"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        },
        {
          "id": "<call_id>",
          "name": "load_skill_resource"
          "status": "success",
          "content": "<tool_call content>",
          "type": "tool_result",
          "round": 1,
        }
      ]
      """
