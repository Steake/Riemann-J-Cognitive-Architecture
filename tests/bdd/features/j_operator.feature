Feature: J-Operator Resolution
  As a cognitive architecture
  I need to resolve high prediction error
  So that symbolic failures are handled gracefully

  Scenario: J-Shift triggers on high PN
    Given a prediction error signal with PN 0.95
    When the J-Operator processes the signal
    Then a J-Shift should be triggered
    And the system should enter latent space

  Scenario: J-Operator converges with adaptive LR
    Given a J-Shift in progress
    When the operator iterates toward convergence
    Then the learning rate should adapt based on distance
    And convergence should be reached within max iterations

  Scenario: Lyapunov stability analysis
    Given a converged J-Operator trajectory
    When stability is analyzed
    Then a Lyapunov exponent should be calculated
    And the result should indicate convergence stability

  Scenario: J-Operator respects iteration limits
    Given a J-Operator with difficult convergence
    When max iterations are reached
    Then the operator should stop
    And return a status of "ITER_LIMIT_EXCEEDED"
