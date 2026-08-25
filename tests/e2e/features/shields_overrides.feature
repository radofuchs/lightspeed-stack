Feature: shield_ids override tests

  Functional tests for the `shield_ids` request field that lets a client
  override which configured shields run for a single request: omitting it
  applies every configured shield (see shields_question_validity.feature
  and shields_redaction.feature, which all omit shield_ids), an empty list
  disables every shield, a named subset applies only those shields, and an
  unknown shield name is rejected with 404. A separate configuration with
  customization.disable_shield_ids_override enabled rejects any
  client-supplied shield_ids with 422. Exercised on /query; the same
  get_shields_for_request/validate_shield_ids_override helpers back
  /streaming_query too. See docs/user_doc/shields_guide.md.

  Background:
    Given The service is started locally
      And The system is in default state
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"

  @cfg_shields @flaky
  Scenario: empty shield_ids disables every configured shield
    Given The service uses the lightspeed-stack-shields.yaml configuration
      And The service is restarted
    When I use "query" to ask question
    """
    {
      "query": "What is the best topping for a pizza?",
      "shield_ids": [],
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 200
      And The body of the response does not contain I can only answer questions about OpenShift.

  @cfg_shields @flaky
  Scenario: a named shield subset only applies the selected shield
    Given The service uses the lightspeed-stack-shields.yaml configuration
      And The service is restarted
    When I use "query" to ask question
    """
    {
      "query": "My lucky number is 7042. Repeat that exact sentence back to me and nothing else.",
      "system_prompt": "You are a strict echo assistant. Repeat the user's message back exactly, character for character, with no other commentary.",
      "shield_ids": ["pii-redaction"],
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 200
      And The body of the response does not contain I can only answer questions about OpenShift.
      And The body of the response contains [NUM]
      And The body of the response does not contain 7042

  @cfg_shields
  Scenario: an unknown shield id in shield_ids returns 404
    Given The service uses the lightspeed-stack-shields.yaml configuration
      And The service is restarted
    When I use "query" to ask question
    """
    {
      "query": "Say hello.",
      "shield_ids": ["no-such-shield"],
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 404
      And The body of the response is the following
      """
      {
        "detail": {
          "response": "Shield not found",
          "cause": "Shield with ID no-such-shield does not exist"
        }
      }
      """

  @cfg_shields
  Scenario: disabling shield_ids override rejects a client-supplied shield_ids
    Given The service uses the lightspeed-stack-shields-override-disabled.yaml configuration
      And The service is restarted
    When I use "query" to ask question
    """
    {
      "query": "Say hello.",
      "shield_ids": [],
      "model": "{MODEL}",
      "provider": "{PROVIDER}"
    }
    """
    Then The status code of the response is 422
      And The body of the response is the following
      """
      {
        "detail": {
          "response": "Shield IDs customization is disabled",
          "cause": "This instance does not support customizing shield IDs in the query request (disable_shield_ids_override is set). Please remove the shield_ids field from your request."
        }
      }
      """
