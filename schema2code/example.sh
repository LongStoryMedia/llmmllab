#!/bin/bash

# Make the schema2code script executable
chmod +x schema2code.py

# Example usages with the Model schema
echo "Generating Go types..."
./schema2code.py ../maistro/models/schema.json --language go --output ../maistro/models/generated_model.go --package models

echo "Generating Python types..."
./schema2code.py ../maistro/models/schema.json --language python --output ../maistro/models/generated_model.py

echo "Generating TypeScript types..."
./schema2code.py ../maistro/models/schema.json --language typescript --output ../maistro/models/generated_model.ts

echo "Generating C# types..."
./schema2code.py ../maistro/models/schema.json --language csharp --output ../maistro/models/GeneratedModel.cs --namespace Maistro.Models

echo "All done! Check the generated files."
