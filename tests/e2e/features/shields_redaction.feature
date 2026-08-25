Feature: redaction shield functional tests

  Functional tests for the LCORE-owned `redaction` shield: text matching a
  configured pattern (digits, per lightspeed-stack-shields.yaml) is
  substituted with the replacement token in both directions (input to the
  model, output back to the caller); text with no match passes through
  unchanged. The system prompt instructs the model to echo the user's
  message verbatim so the redaction (or lack thereof) is directly
  observable in the response. Queries stay on-topic (mention OpenShift) so
  the `question_validity` shield configured alongside `redaction` does not
  short-circuit the turn. See docs/user_doc/shields_guide.md.

  Background:
    Given The service is started locally
      And The system is in default state
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"
      And The service uses the lightspeed-stack-shields.yaml configuration
      And The service is restarted

  @cfg_shields @flaky
  Scenario: query endpoint redacts matching PII in the response
    When I use "query" to ask question
    """
    {
      "query": "My OpenShift support ticket number is 48213. Repeat that exact sentence back to me and nothing else.",
      "system_prompt": "You are a strict echo assistant. Repeat the user's message back exactly, character for character, with no other commentary.",
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 200
      And The body of the response contains [NUM]
      And The body of the response does not contain 48213

  @cfg_shields @flaky
  Scenario: query endpoint leaves non-matching text unchanged
    When I use "query" to ask question
    """
    {
      "query": "My OpenShift cluster is healthy and stable. Repeat that exact sentence back to me and nothing else.",
      "system_prompt": "You are a strict echo assistant. Repeat the user's message back exactly, character for character, with no other commentary.",
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 200
      And The response contains following fragments
          | Fragments in LLM response                     |
          | My OpenShift cluster is healthy and stable.   |
      And The body of the response does not contain [NUM]

  @cfg_shields @flaky
  Scenario: streaming_query endpoint redacts matching PII in the response
    When I use "streaming_query" to ask question
    """
    {
      "query": "My OpenShift support ticket number is 48213. Repeat that exact sentence back to me and nothing else.",
      "system_prompt": "You are a strict echo assistant. Repeat the user's message back exactly, character for character, with no other commentary.",
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The body of the response contains [NUM]
      And The body of the response does not contain 48213

  @cfg_shields @flaky
  Scenario: streaming_query endpoint leaves non-matching text unchanged
    When I use "streaming_query" to ask question
    """
    {
      "query": "My OpenShift cluster is healthy and stable. Repeat that exact sentence back to me and nothing else.",
      "system_prompt": "You are a strict echo assistant. Repeat the user's message back exactly, character for character, with no other commentary.",
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The streamed response contains following fragments
          | Fragments in LLM response                     |
          | My OpenShift cluster is healthy and stable.   |
      And The body of the response does not contain [NUM]
