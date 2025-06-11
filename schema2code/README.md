# schema2code

A tool for converting JSON schema files to type definitions in various programming languages.

## Supported languages

- Go
- Python (with Pydantic or dataclasses)
- TypeScript
- C# (.NET)

## Installation

```bash
# Make the script executable
chmod +x schema2code.py

# Optional: Create a symbolic link to make it available system-wide
sudo ln -s "$(pwd)/schema2code.py" /usr/local/bin/schema2code
```

## Usage

```bash
python schema2code.py [schema_file] --language [language] --output [output_file] [options]
```

### Required arguments:

- `schema_file`: Path to the JSON schema file
- `--language`, `-l`: Target language (`go`, `python`, `typescript`, `csharp`, `dotnet`)
- `--output`, `-o`: Output file path

### Optional arguments:

- `--mode`: `create` (default) or `append` - Whether to create a new file or append to existing
- `--no-create`: Don't create the file if it doesn't exist
- `--package`: Go package name (default: "main")
- `--namespace`: C# namespace (default: "SchemaTypes")
- `--no-pydantic`: Use dataclasses instead of Pydantic for Python
- `--no-overwrite`: Prevents overwriting of existing files

## Examples

Generate Go types:
```bash
python schema2code.py schema.json --language go --output models.go --package models
```

Generate Python types with Pydantic:
```bash
python schema2code.py schema.json --language python --output models.py
```

Generate Python types with dataclasses:
```bash
python schema2code.py schema.json --language python --output models.py --no-pydantic
```

Generate TypeScript interfaces:
```bash
python schema2code.py schema.json --language typescript --output models.ts
```

Generate C# classes:
```bash
python schema2code.py schema.json --language csharp --output Models.cs --namespace MyApp.Models
```

Append to existing file:
```bash
python schema2code.py schema.json --language go --output models.go --mode append
```
