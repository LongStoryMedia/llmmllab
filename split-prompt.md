# Decoupling Effort

## Skills
/context-degredation
/context-optimization

## Goal

The goal of this decoupling effort is to separate the Runner, Server, and Composer components of the llmmllab project into distinct and independently deployable services. This will allow for greater flexibility, scalability, and maintainability of the system.

## Tasks

@"architect-refactor-expert (agent)" and @"microservices-architect (agent)"
- proto will be a repository that contains the protocol buffer definitions for the gRPC communication between the Runner, Server, and Composer. This will allow for clear and consistent communication between the components.
  - the name of the repo should follow the pattern "llmmllab-proto" to maintain consistency with the existing naming conventions.
- Server should contain manifest files for the postgres database as well. it currently uses the manifests at "../k3s-cluster/psql" which can be largely replicated, but in the llmmll namespace. This will allow for better organization and separation of concerns.
- Each component should have its own Makefile for building, testing, and deploying. This will allow for greater flexibility and independence of each component.
- scripts to generate models will be moved into the schemas repository. This will allow for better organization and separation of concerns.
  - (Note: most scripts are in the root of this directory. They will need to be moved, modified, and/or deleted)

@"test-strategy-architect (agent)"
- tests will be organized into separate directories for each component (Runner, Server, Composer) to ensure that tests are focused and relevant to each component's functionality.
  - Server will contain integration tests to test integration between the Server and Composer.
  - Composer will contain integration tests to test integration between the Composer and Runner.
  - UI will contain end-to-end tests to test the entire flow from the UI to the Runner.
  - Server, Composer, Runner, and UI will each contain unit tests to test the individual functionality of each component.