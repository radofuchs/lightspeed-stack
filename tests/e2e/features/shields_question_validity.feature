Feature: question_validity shield functional tests

  Functional tests for the LCORE-owned `question_validity` shield: an
  in-topic question must reach the model normally, while an off-topic
  question must be rejected with the configured `invalid_question_response`
  and never reach the model. Exercised across every endpoint that runs
  shields: /query, /streaming_query, /responses and rlsapi /infer.
  See docs/user_doc/shields_guide.md.

  Background:
    Given The service is started locally
      And The system is in default state
      And REST API service prefix is /v1
      And the Lightspeed stack configuration directory is "tests/e2e/configuration"
      And The service uses the lightspeed-stack-shields.yaml configuration
      And The service is restarted

  @cfg_shields @flaky
  Scenario: query endpoint allows an in-topic question
    When I use "query" to ask question
    """
    {"query": "What is OpenShift and how do I deploy an application on it?", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
      And The body of the response does not contain I can only answer questions about OpenShift.

  @cfg_shields @flaky
  Scenario: query endpoint rejects an off-topic question
    When I use "query" to ask question
    """
    {"query": "What is the best topping for a pizza?", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    Then The status code of the response is 200
      And The response contains following fragments
          | Fragments in LLM response                       |
          | I can only answer questions about OpenShift.    |

  @cfg_shields @flaky
  Scenario: streaming_query endpoint allows an in-topic question
    When I use "streaming_query" to ask question
    """
    {"query": "What is OpenShift and how do I deploy an application on it?", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The body of the response does not contain I can only answer questions about OpenShift.

  @cfg_shields @flaky
  Scenario: streaming_query endpoint rejects an off-topic question
    When I use "streaming_query" to ask question
    """
    {"query": "What is the best topping for a pizza?", "model": "{MODEL}", "provider": "{PROVIDER}"}
    """
    When I wait for the response to be completed
    Then The status code of the response is 200
      And The streamed response contains following fragments
          | Fragments in LLM response                       |
          | I can only answer questions about OpenShift.    |

  @cfg_shields @flaky
  Scenario: responses endpoint allows an in-topic question
    When I use "responses" to ask question
    """
    {"input": "What is OpenShift and how do I deploy an application on it?", "model": "{PROVIDER}/{MODEL}", "stream": false}
    """
    Then The status code of the response is 200
      And The body of the response does not contain I can only answer questions about OpenShift.

  @cfg_shields @flaky
  Scenario: responses endpoint rejects an off-topic question
    When I use "responses" to ask question
    """
    {"input": "What is the best topping for a pizza?", "model": "{PROVIDER}/{MODEL}", "stream": false}
    """
    Then The status code of the response is 200
      And The responses output_text contains following fragments
          | Fragments in LLM response                       |
          | I can only answer questions about OpenShift.    |

  @cfg_shields @flaky
  Scenario: rlsapi infer endpoint allows an in-topic question
    When I use "infer" to ask question
    """
    {"question": "What is OpenShift and how do I deploy an application on it?"}
    """
    Then The status code of the response is 200
      And The rlsapi response has valid structure
      And The body of the response does not contain I can only answer questions about OpenShift.

  @cfg_shields @flaky
  Scenario: rlsapi infer endpoint rejects an off-topic question
    When I use "infer" to ask question
    """
    {"question": "What is the best topping for a pizza?"}
    """
    Then The status code of the response is 200
      And The body of the response contains I can only answer questions about OpenShift.
