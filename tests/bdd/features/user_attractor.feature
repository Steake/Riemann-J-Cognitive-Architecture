Feature: User Attractor System
  As a cognitive architecture
  I need to maintain unique user models
  So that I can form persistent relationships

  Scenario: Create new user attractor
    Given a cognitive workspace
    When a new user "alice" interacts
    Then a UserAttractor should be created for "alice"
    And the attractor should have an empty state history

  Scenario: Build user state history
    Given a UserAttractor for "bob"
    When user states are added over time
    Then the state history should grow
    And the GMM should be updated periodically

  Scenario: Apply affinity to user input
    Given a trained UserAttractor for "charlie"
    When a new user state is processed
    Then the state should be modified by affinity
    And the modification should pull toward user centroids

  Scenario: Multi-user state isolation
    Given multiple users "dave" and "eve"
    When both users interact with the system
    Then each should have a separate attractor
    And state histories should not overlap
