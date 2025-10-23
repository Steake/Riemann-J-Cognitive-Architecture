Feature: Prediction Error Driver
  As a cognitive architecture
  I need a continuous source of prediction error
  So that the system maintains dynamic equilibrium

  Scenario: PN Driver starts with low prediction error
    Given a new PN Driver instance
    When the driver is initialized
    Then the prediction error should be low
    And the current t value should be at the first zero

  Scenario: PN increases with computation steps
    Given a PN Driver with zero steps
    When computation steps are accumulated
    Then the prediction error should increase
    And follow a sigmoid curve

  Scenario: PN resets when zero is found
    Given a PN Driver with high step count
    When a Riemann zero is found
    Then the step count should reset to zero
    And the prediction error should decrease

  Scenario: PN signals are added to workspace
    Given a running PN Driver
    When prediction error is calculated
    Then a PredictionErrorSignal should be created
    And the signal should be added to the global workspace
    And the signal priority should reflect PN value
