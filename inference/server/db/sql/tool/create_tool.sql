-- Create a new dynamic tool
INSERT INTO dynamic_tools(user_id, name, description, code, function_name, embedding, parameters)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING
    id, user_id, name, description, code, function_name, embedding, parameters, created_at, updated_at;

