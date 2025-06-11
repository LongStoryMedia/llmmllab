# SQL Queries Organization

This directory contains SQL queries used throughout the proxyllama application, organized by domain.

## Directory Structure

Each subdirectory represents a domain or functional area of the application:

- `conversation/`: Queries related to conversation operations
- `embedding/`: Queries related to embedding creation and retrieval
- `memory/`: Queries related to memory and search operations
- `message/`: Queries related to message operations
- `modelprofile/`: Queries related to model profile management
- `research/`: Queries related to research tasks and subtasks
- `summary/`: Queries related to conversation summaries
- `user/`: Queries related to user management

## Query Naming Convention

Queries are named according to their purpose, following the pattern:

```
[operation]_[entity].sql
```

For example:
- `get_message.sql`: Query to retrieve a single message
- `create_summary.sql`: Query to create a new summary
- `update_config.sql`: Query to update user configuration

## How to Use

Queries are loaded automatically at application startup using the embedded filesystem.
To use queries in code, access them through the `GetQuery()` function:

```go
// Example usage
func GetMessageByID(ctx context.Context, messageID int) (*Message, error) {
    var message Message
    err := Pool.QueryRow(ctx, GetQuery("message.get_message"), messageID).Scan(
        &message.ID, &message.ConversationID, &message.Role, &message.Content, &message.CreatedAt)
    // ...
}
```

## Adding New Queries

To add a new query:

1. Create a new SQL file in the appropriate subdirectory
2. Use descriptive comments at the top of the file to explain the query's purpose
3. Name the query file according to the naming convention
4. The file will be automatically loaded at application startup

## Benefits of This Approach

- **Separation of concerns**: SQL logic is separate from application logic
- **Maintainability**: Easier to find and update SQL statements
- **Readability**: SQL queries are properly formatted and commented
- **Reusability**: Queries can be reused across different parts of the application
- **Version control**: Changes to queries can be tracked in version control
- **Performance**: Queries are loaded once at startup and cached