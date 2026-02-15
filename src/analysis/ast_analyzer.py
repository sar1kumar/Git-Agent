"""Tree-sitter based AST analyzer for accurate code parsing."""

import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function extracted from AST."""
    name: str
    start_line: int
    end_line: int
    parameters: list[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    docstring: Optional[str] = None
    body_start_line: int = 0
    complexity: int = 1  # Base complexity
    nested_functions: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class extracted from AST."""
    name: str
    start_line: int
    end_line: int
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ImportInfo:
    """Information about imports."""
    module: str
    names: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    line: int = 0
    is_from_import: bool = False


@dataclass
class ASTAnalysis:
    """Complete AST analysis result."""
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    global_variables: list[tuple[str, int]] = field(default_factory=list)
    complexity_score: int = 0
    loc: int = 0
    sloc: int = 0  # Source lines of code (excluding comments/blanks)


class ASTAnalyzer:
    """Analyzer using Tree-sitter for accurate AST parsing."""
    
    # Complexity-increasing node types
    COMPLEXITY_NODES = {
        'if_statement', 'elif_clause', 'for_statement', 'while_statement',
        'except_clause', 'with_statement', 'assert_statement',
        'boolean_operator', 'conditional_expression', 'list_comprehension',
        'dictionary_comprehension', 'set_comprehension', 'generator_expression',
        'match_statement', 'case_clause',
    }
    
    def __init__(self):
        """Initialize the AST analyzer."""
        self.parser: Optional[Parser] = None
        self._setup_parser()
    
    def _setup_parser(self) -> None:
        """Set up the Tree-sitter parser."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. Install with: pip install tree-sitter tree-sitter-python")
            return
        
        try:
            self.parser = Parser(Language(tspython.language()))
            logger.info("Tree-sitter parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter parser: {e}")
            self.parser = None
    
    @property
    def is_available(self) -> bool:
        """Check if AST analysis is available."""
        return self.parser is not None
    
    def analyze(self, code: str, filename: str = "unknown.py") -> ASTAnalysis:
        """
        Analyze code and extract AST information.
        
        Args:
            code: Source code to analyze.
            filename: Name of the file (for language detection).
            
        Returns:
            ASTAnalysis with extracted information.
        """
        if not self.is_available:
            return self._fallback_analyze(code)
        
        # Determine language from filename
        ext = Path(filename).suffix.lower()
        if ext != '.py':
            logger.debug(f"Unsupported file type: {ext}, using fallback")
            return self._fallback_analyze(code)
        
        try:
            tree = self.parser.parse(bytes(code, 'utf-8'))
            return self._analyze_tree(tree.root_node, code)
        except Exception as e:
            logger.error(f"AST analysis failed: {e}")
            return self._fallback_analyze(code)
    
    def _analyze_tree(self, root: "Node", code: str) -> ASTAnalysis:
        """Analyze the parsed tree."""
        analysis = ASTAnalysis()
        lines = code.split('\n')
        analysis.loc = len(lines)
        analysis.sloc = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
        # Walk the tree
        self._walk_node(root, analysis, code)
        
        # Calculate total complexity
        analysis.complexity_score = sum(f.complexity for f in analysis.functions)
        for cls in analysis.classes:
            analysis.complexity_score += sum(m.complexity for m in cls.methods)
        
        return analysis
    
    def _walk_node(self, node: "Node", analysis: ASTAnalysis, code: str, parent_class: Optional[ClassInfo] = None) -> None:
        """Recursively walk AST nodes."""
        if node.type == 'function_definition':
            func = self._extract_function(node, code)
            if parent_class:
                parent_class.methods.append(func)
            else:
                analysis.functions.append(func)
        
        elif node.type == 'class_definition':
            cls = self._extract_class(node, code)
            analysis.classes.append(cls)
            # Don't recurse into class children here, handled in _extract_class
            return
        
        elif node.type == 'import_statement':
            imp = self._extract_import(node, code)
            analysis.imports.append(imp)
        
        elif node.type == 'import_from_statement':
            imp = self._extract_from_import(node, code)
            analysis.imports.append(imp)
        
        elif node.type == 'expression_statement' and parent_class is None:
            # Check for global variable assignments
            if len(node.children) > 0 and node.children[0].type == 'assignment':
                var_name = self._get_node_text(node.children[0].children[0], code)
                if var_name and not var_name.startswith('_'):
                    analysis.global_variables.append((var_name, node.start_point[0] + 1))
        
        # Recurse into children
        for child in node.children:
            self._walk_node(child, analysis, code, parent_class)
    
    def _extract_function(self, node: "Node", code: str) -> FunctionInfo:
        """Extract function information from AST node."""
        func = FunctionInfo(
            name="",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
        
        for child in node.children:
            if child.type == 'identifier':
                func.name = self._get_node_text(child, code)
            elif child.type == 'parameters':
                func.parameters = self._extract_parameters(child, code)
            elif child.type == 'type':
                func.return_type = self._get_node_text(child, code)
            elif child.type == 'block':
                func.body_start_line = child.start_point[0] + 1
                func.complexity = self._calculate_complexity(child)
                func.calls = self._extract_calls(child, code)
                # Check for docstring
                if len(child.children) > 0:
                    first_stmt = child.children[0]
                    if first_stmt.type == 'expression_statement':
                        if len(first_stmt.children) > 0 and first_stmt.children[0].type == 'string':
                            func.docstring = self._get_node_text(first_stmt.children[0], code)
            elif child.type == 'decorator':
                func.decorators.append(self._get_node_text(child, code))
        
        # Check if async
        func.is_async = any(c.type == 'async' for c in node.children)
        
        return func
    
    def _extract_class(self, node: "Node", code: str) -> ClassInfo:
        """Extract class information from AST node."""
        cls = ClassInfo(
            name="",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
        
        for child in node.children:
            if child.type == 'identifier':
                cls.name = self._get_node_text(child, code)
            elif child.type == 'argument_list':
                # Base classes
                for arg in child.children:
                    if arg.type == 'identifier':
                        cls.bases.append(self._get_node_text(arg, code))
            elif child.type == 'block':
                # Extract methods and docstring
                for block_child in child.children:
                    if block_child.type == 'function_definition':
                        method = self._extract_function(block_child, code)
                        cls.methods.append(method)
                    elif block_child.type == 'expression_statement' and not cls.docstring:
                        if len(block_child.children) > 0 and block_child.children[0].type == 'string':
                            cls.docstring = self._get_node_text(block_child.children[0], code)
            elif child.type == 'decorator':
                cls.decorators.append(self._get_node_text(child, code))
        
        return cls
    
    def _extract_import(self, node: "Node", code: str) -> ImportInfo:
        """Extract import statement information."""
        imp = ImportInfo(module="", line=node.start_point[0] + 1)
        
        for child in node.children:
            if child.type == 'dotted_name':
                imp.module = self._get_node_text(child, code)
            elif child.type == 'aliased_import':
                imp.module = self._get_node_text(child.children[0], code)
                if len(child.children) > 1:
                    imp.alias = self._get_node_text(child.children[-1], code)
        
        return imp
    
    def _extract_from_import(self, node: "Node", code: str) -> ImportInfo:
        """Extract from...import statement information."""
        imp = ImportInfo(module="", line=node.start_point[0] + 1, is_from_import=True)
        
        for child in node.children:
            if child.type == 'dotted_name':
                imp.module = self._get_node_text(child, code)
            elif child.type == 'import_from_names':
                for name_node in child.children:
                    if name_node.type in ('identifier', 'dotted_name'):
                        imp.names.append(self._get_node_text(name_node, code))
            elif child.type == 'wildcard_import':
                imp.names.append('*')
        
        return imp
    
    def _extract_parameters(self, node: "Node", code: str) -> list[str]:
        """Extract function parameters."""
        params = []
        for child in node.children:
            if child.type == 'identifier':
                params.append(self._get_node_text(child, code))
            elif child.type in ('default_parameter', 'typed_parameter', 'typed_default_parameter'):
                # Get the parameter name
                for sub in child.children:
                    if sub.type == 'identifier':
                        params.append(self._get_node_text(sub, code))
                        break
            elif child.type in ('list_splat_pattern', 'dictionary_splat_pattern'):
                for sub in child.children:
                    if sub.type == 'identifier':
                        prefix = '*' if child.type == 'list_splat_pattern' else '**'
                        params.append(prefix + self._get_node_text(sub, code))
                        break
        return params
    
    def _extract_calls(self, node: "Node", code: str) -> list[str]:
        """Extract function calls from a code block."""
        calls = []
        self._find_calls(node, code, calls)
        return list(set(calls))
    
    def _find_calls(self, node: "Node", code: str, calls: list[str]) -> None:
        """Recursively find function calls."""
        if node.type == 'call':
            # Get the function being called
            if len(node.children) > 0:
                func_node = node.children[0]
                if func_node.type == 'identifier':
                    calls.append(self._get_node_text(func_node, code))
                elif func_node.type == 'attribute':
                    calls.append(self._get_node_text(func_node, code))
        
        for child in node.children:
            self._find_calls(child, code, calls)
    
    def _calculate_complexity(self, node: "Node") -> int:
        """Calculate cyclomatic complexity of a code block."""
        complexity = 1  # Base complexity
        
        def count_complexity(n: "Node"):
            nonlocal complexity
            if n.type in self.COMPLEXITY_NODES:
                complexity += 1
            for child in n.children:
                count_complexity(child)
        
        count_complexity(node)
        return complexity
    
    def _get_node_text(self, node: "Node", code: str) -> str:
        """Get the text content of a node."""
        return code[node.start_byte:node.end_byte]
    
    def _fallback_analyze(self, code: str) -> ASTAnalysis:
        """Fallback analysis when Tree-sitter is not available."""
        import re
        
        analysis = ASTAnalysis()
        lines = code.split('\n')
        analysis.loc = len(lines)
        analysis.sloc = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
        # Simple regex-based extraction
        func_pattern = re.compile(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)\s*[\(:]')
        import_pattern = re.compile(r'^(from\s+(\S+)\s+)?import\s+(.+)')
        
        for i, line in enumerate(lines, start=1):
            func_match = func_pattern.match(line)
            if func_match:
                func = FunctionInfo(
                    name=func_match.group(3),
                    start_line=i,
                    end_line=i,
                    is_async=bool(func_match.group(2)),
                )
                analysis.functions.append(func)
            
            class_match = class_pattern.match(line)
            if class_match:
                cls = ClassInfo(
                    name=class_match.group(2),
                    start_line=i,
                    end_line=i,
                )
                analysis.classes.append(cls)
            
            import_match = import_pattern.match(line.strip())
            if import_match:
                imp = ImportInfo(
                    module=import_match.group(2) or import_match.group(3).split()[0],
                    line=i,
                    is_from_import=bool(import_match.group(1)),
                )
                analysis.imports.append(imp)
        
        return analysis
    
    def get_function_at_line(self, analysis: ASTAnalysis, line: int) -> Optional[FunctionInfo]:
        """Get the function containing a specific line."""
        for func in analysis.functions:
            if func.start_line <= line <= func.end_line:
                return func
        
        for cls in analysis.classes:
            for method in cls.methods:
                if method.start_line <= line <= method.end_line:
                    return method
        
        return None
    
    def get_class_at_line(self, analysis: ASTAnalysis, line: int) -> Optional[ClassInfo]:
        """Get the class containing a specific line."""
        for cls in analysis.classes:
            if cls.start_line <= line <= cls.end_line:
                return cls
        return None
    
    def find_complex_functions(self, analysis: ASTAnalysis, threshold: int = 10) -> list[FunctionInfo]:
        """Find functions exceeding complexity threshold."""
        complex_funcs = []
        
        for func in analysis.functions:
            if func.complexity > threshold:
                complex_funcs.append(func)
        
        for cls in analysis.classes:
            for method in cls.methods:
                if method.complexity > threshold:
                    complex_funcs.append(method)
        
        return complex_funcs
    
    def find_long_functions(self, analysis: ASTAnalysis, threshold: int = 50) -> list[FunctionInfo]:
        """Find functions exceeding line count threshold."""
        long_funcs = []
        
        for func in analysis.functions:
            length = func.end_line - func.start_line + 1
            if length > threshold:
                long_funcs.append(func)
        
        for cls in analysis.classes:
            for method in cls.methods:
                length = method.end_line - method.start_line + 1
                if length > threshold:
                    long_funcs.append(method)
        
        return long_funcs
