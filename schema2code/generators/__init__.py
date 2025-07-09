# This is the init file for the generators
from generators.dotnet import CSharpGenerator
from generators.go import GoGenerator
from generators.python import PythonGenerator
from generators.typescript import TypeScriptGenerator
from generators.proto import ProtoGenerator

__all__ = ['CSharpGenerator', 'GoGenerator', 'PythonGenerator', 'TypeScriptGenerator', 'ProtoGenerator']
