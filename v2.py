import os
import shutil
import subprocess

def main():
    build("example_s2", "example")
    # transpile_module("example/main.s2", "out")


def build(dir, prog_name):
    # Parse modules
    modules = {}
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            module = open_and_parse_module(filepath)
            modules[module.name] = module

    print("Modules parsed: " + ", ".join(module for module in modules))

    for name, module in modules.items():
        # if name not in {'main', 'supermath'}:
        print(f"Resolving names in {name}")
        resolve_names(module, modules)

    print("AST: ")
    print_ast(list(modules.values()))

    # Codegen modules into C files
    out_path = os.path.join(dir, "c_out")
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)
    
    for module in modules.values():
        gen_and_write_module(module, out_path)

    # Build C program
    # bin_path = os.path.join(dir, "bin")
    # shutil.rmtree(bin_path, ignore_errors=True)
    # os.makedirs(bin_path)

    # cmd = ["clang"]
    # for module in modules:
    #     cmd.append(f'{out_path}/{module}.c')
    # cmd.append("-o")
    # cmd.append(f'{bin_path}/{prog_name}')
    # print(str.join(" ", cmd))
    # subprocess.run(cmd, check=True)

def open_and_parse_module(filepath):
    print(f"Parsing module {filepath}")
    with open(filepath) as f:
        src = f.read()
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    # print(tokens)
    parser = Parser(tokens)
    module = parser.parse_module()
    return module

def gen_and_write_module(module, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{module.name}.h"), "w") as h:
        header = gen_c_header(module) 
        h.write(header)
    with open(os.path.join(out_dir, f"{module.name}.c"), "w") as c:
        source = gen_c_source(module)
        c.write(source)

### LEXING

class Token:
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


class Lexer:
    def __init__(self, src):
        self.src = src
        self.pos = 0
        self.len = len(src)

    def peek(self):
        return self.src[self.pos] if self.pos < self.len else '\0'

    def advance(self):
        ch = self.peek()
        self.pos += 1
        return ch

    def skip_whitespace(self):
        while self.peek() in ' \r\t\n':
            self.advance()

    def match_keyword_or_indent(self, first):
        ident = first
        while self.peek().isalnum() or self.peek() == '_':
            ident += self.advance()
        KEYWORDS = {
            'return',
            'module',
            'import',
            'extern',
            'from'
        }
        if ident in KEYWORDS:
            return Token(ident.upper(), ident)
        return Token('IDENT', ident)

    def match_number(self, first):
        num = first
        while self.peek().isdigit():
            num += self.advance()
        return Token('NUMBER', num)

    def read_string(self):
        result = ''
        while True:
            ch = self.advance()
            if ch == '"':
                break
            result += ch
        return Token('STRING', result)

    def next_token(self):
        self.skip_whitespace()
        ch = self.advance()

        if ch.isalpha() or ch == '_':
            return self.match_keyword_or_indent(ch)
        
        if ch.isdigit():
            return self.match_number(ch)

        if ch == '"':
            return self.read_string()
        
        SINGLE_CH_TOKENS = {
            '(': 'LPAREN',
            ')': 'RPAREN',
            '{': 'LBRACE',
            '}': 'RBRACE',
            ',': 'COMMA',
            ';': 'SEMICOLON',
            '+': 'PLUS',
            '=': 'EQUAL',
            '<': 'LT',
            '>': 'GT',
            '.': 'PERIOD',
            '#': 'HASH',
            '*': 'STAR'
        }

        if ch in SINGLE_CH_TOKENS:
            return Token(SINGLE_CH_TOKENS[ch], ch)
        
        if ch == '\0':
            return Token('EOF', '')
        
        raise SyntaxError(f'Unexpected character: {ch}')
    
    def tokenize(self):
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.kind == 'EOF':
                break
        return tokens

### PARSING

class Node:
    pass

class Expr(Node):
    pass

class Stmt(Node):
    pass

class Number(Expr):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Expr:Number({self.value})"

class String(Expr):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Expr:String({self.value})"

class Identifier(Expr):
    def __init__(self, name, resolved=None):
        self.name = name
        self.resolved = resolved

    def __repr__(self):
        return f"Expr:Identifier({self.name})"

class BinaryOp(Expr):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"Expr:BinaryOp('{self.left}', '{self.op}', '{self.right}')"

class Call(Expr):
    def __init__(self, func_name, args, resolved=None):
        self.func_name = func_name
        self.args = args
        self.resolved = resolved
    
    def __repr__(self):
        return f"Expr:Call({self.func_name}, {self.args})"

class Return(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Stmt:Return({self.expr})"

class VarDecl(Stmt):
    def __init__(self, var_type, name, expr):
        self.var_type = var_type
        self.name = name
        self.expr = expr
    
    def __repr__(self):
        return f"Stmt:VarDecl({self.var_type}, {self.name}, {self.expr})"

class ExprStmt(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Stmt:ExprStmt({self.expr})"

class Function(Node):
    def __init__(self, return_type, name, params, body):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body

    def __repr__(self):
        return f"Function({self.return_type}, {self.name}, {self.params}, {self.body}"

class ExternFunc(Node):
    def __init__(self, return_type, name, param_types):
        self.return_type = return_type
        self.name = name
        self.param_types = param_types

    def __repr__(self):
        return f"ExternFunc({self.return_type}, {self.name}, {self.param_types})"

class ExternBlock(Node):
    def __init__(self, header_name, functions):
        self.header_name = header_name
        self.functions = functions

    def __repr__(self):
        return f"ExternBlock({self.header_name}, {self.functions})"

class Module(Node):
    def __init__(self, name, imports, extern_blocks, funcs):
        self.name = name
        self.imports = imports
        self.extern_blocks = extern_blocks
        self.funcs = funcs

    def __repr__(self):
        return f"Module({self.name}, {self.imports}, {self.extern_blocks}, {self.funcs})"

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def peek2(self):
        return self.tokens[self.pos + 1]

    def advance(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, kind, optional=False):
        token = self.peek()
        if token.kind != kind and not optional:
            raise SyntaxError(f"Expected {kind}, got {token.kind}");
        if token.kind == kind:
            self.advance()
            return token
        else:
            return None

    def parse_module(self):
        self.expect('MODULE')
        name = self.expect('IDENT').value
        self.expect('SEMICOLON')

        imports = []
        extern_blocks = []
        funcs = []

        while self.peek().kind != 'EOF':
            if self.peek().kind == 'IMPORT':
                self.advance()
                imports.append(self.expect('IDENT').value)
                self.expect('SEMICOLON')
            elif self.peek().kind == 'EXTERN':
                block = self.parse_extern_block()
                extern_blocks.append(block)
            else:
                funcs.append(self.parse_function())

        return Module(name, imports, extern_blocks, funcs)

    def parse_extern_block(self):
        self.expect('EXTERN')
        self.expect('FROM')
        header = self.expect('STRING').value
        self.expect('LBRACE')
        funcs = []
        while self.peek().kind != 'RBRACE':
            ret_type = self.expect('IDENT').value
            name = self.expect('IDENT').value
            self.expect('LPAREN')
            param_types = []
            if self.peek().kind != 'RPAREN':
                while True:
                    typ = self.expect('IDENT').value
                    self.expect('STAR', optional=True)
                    self.expect('IDENT') # skip param name
                    param_types.append(typ)
                    if self.peek().kind == 'COMMA':
                        self.advance()
                    else:
                        break
            self.expect('RPAREN')
            self.expect('SEMICOLON')
            funcs.append(ExternFunc(ret_type, name, param_types))
        self.expect('RBRACE')
        return ExternBlock(header, funcs)

    def parse_function(self):
        ret_type = self.expect('IDENT').value
        name = self.expect('IDENT').value
        self.expect('LPAREN')

        params = []
        while self.peek().kind != 'RPAREN':
            type_tok = self.expect('IDENT')
            name_tok = self.expect('IDENT')
            params.append((type_tok.value, name_tok.value))
            if self.peek().kind == 'COMMA':
                self.advance()
        self.expect('RPAREN')
        self.expect('LBRACE')

        body = self.parse_block()

        self.expect('RBRACE')
        return Function(ret_type, name, params, body)

    def parse_block(self):
        stmts = []
        while self.peek().kind != 'RBRACE':
            tok = self.peek()
            if tok.kind == 'RETURN':
                stmts.append(self.parse_return())
            elif tok.kind == 'IDENT':
                if self.peek2().kind == 'IDENT':
                    stmts.append(self.parse_var_decl())
                else:
                    stmts.append(self.parse_expr_stmt())
        return stmts

    def parse_return(self):
        self.expect('RETURN')
        expr = self.parse_expr()
        self.expect('SEMICOLON')
        return Return(expr)

    def parse_var_decl(self):
        var_type = self.expect('IDENT').value
        name = self.expect('IDENT').value
        self.expect('EQUAL')
        expr = self.parse_expr()
        self.expect('SEMICOLON')
        return VarDecl(var_type, name, expr)

    def parse_expr_stmt(self):
        call = self.parse_call()
        self.expect('SEMICOLON')
        return ExprStmt(call)

    def parse_expr(self):
        primary = self.parse_primary()
        if self.peek().kind in {'PLUS'}:
            op = self.advance().value
            right = self.parse_expr()
            return BinaryOp(primary, op, right)
        return primary

    def parse_primary(self):
        tok = self.peek()
        if tok.kind == 'IDENT':
            if self.peek2().kind == 'LPAREN':
                return self.parse_call()
            return Identifier(self.advance().value)
        elif tok.kind == 'NUMBER':
            return Number(self.advance().value)
        elif tok.kind == 'STRING':
            return String(self.advance().value)
        else:
            raise SyntaxError("Expected primary expression")

    def parse_call(self):
        name = self.expect('IDENT').value
        self.expect('LPAREN')
        args = []
        if self.peek().kind != 'RPAREN':
            while True:
                args.append(self.parse_expr())
                if self.peek().kind == 'COMMA':
                    self.advance()
                else:
                    break
        self.expect('RPAREN')
        return Call(name, args)

### AST PRETTY PRINT

def print_ast(node, indent = 0):
    pad = '  ' * indent
    if isinstance(node, list):
        for item in node:
            print_ast(item, indent)
    elif isinstance(node, Module):
        print(f"{pad}Module({node.name})")
        for imp in node.imports:
            print(f"{pad}  Import({imp})")
        for extern_block in node.extern_blocks:
            print(f"{pad}  ExternBlock({extern_block.header_name})")
            for extern_func in extern_block.functions:
                print_ast(extern_func, indent + 2) # ExternBlock is already indent + 1
        for func in node.funcs:
            print_ast(func, indent + 1)
    elif isinstance(node, ExternFunc):
        print(f"{pad}ExternFunc {node.return_type} {node.name}({", ".join(f"{typ}" for typ in node.param_types)})")
    elif isinstance(node, Function):
        print(f"{pad}Function {node.return_type} {node.name}({", ".join(f"{typ} {name}" for typ, name in node.params)})")
        for stmt in node.body:
            print_ast(stmt, indent + 1)
    elif isinstance(node, VarDecl):
        print(f"{pad}VarDecl {node.var_type} {node.name} =")
        print_ast(node.expr, indent + 1)
    elif isinstance(node, Return):
        print(f"{pad}Return")
        print_ast(node.expr, indent + 1)
    elif isinstance(node, ExprStmt):
        print(f"{pad}ExprStmt")
        print_ast(node.expr, indent + 1)
    elif isinstance(node, Call):
        print(f"{pad}Call {node.func_name}(")
        for arg in node.args:
            print_ast(arg, indent + 2)
        print(f"{pad})")
        if node.resolved:
            print(f"{pad} -> resolves to {node.resolved.kind}")
    elif isinstance(node, BinaryOp):
        print(f"{pad}BinaryOp {node.op}")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, Identifier):
        print(f"{pad}Identifer {node.name}")
        if node.resolved:
            print(f"{pad} -> resolves to {node.resolved.kind}")
    elif isinstance(node, Number):
        print(f"{pad}Number {node.value}")
    elif isinstance(node, String):
        print(f"{pad}String {node.value}")
    else:
        print(f"{pad}Unknown {node}")


### NAME RESOLUTION

class Symbol:
    def __init__(self, kind, name, decl=None):
        self.kind = kind
        self.name = name
        self.decl = decl

    def __repr__(self):
        return f"Symbol({self.kind, self.name})"

class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {} # name -> Symbol

    def define(self, kind, name, decl=None):
        if name in self.symbols:
            raise Exception(f"Symbol already declared: {name}")
        self.symbols[name] = Symbol(kind, name, decl);

    def resolve(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.resolve(name)
        else:
            raise Exception(f"Undefined symbol: {name}")

def resolve_names(module, all_modules):
    global_scope = Scope()

    # Imported modules
    for imp in module.imports:
        imported_module = all_modules[imp]
        imported_scope = build_exported_scope(imported_module)
        for _, symbol in imported_scope.symbols.items():
            global_scope.define(symbol.kind, symbol.name, symbol.decl)

    # Extern blocks (C includes)
    for block in module.extern_blocks:
        for func in block.functions:
            global_scope.define('func', func.name, func)

    # Module-local functions
    for func in module.funcs:
        global_scope.define('func', func.name, func)

    # Resolve names in every statement inside every function
    for func in module.funcs:
        func_scope = Scope(global_scope)
        for _, param_name in func.params:
            func_scope.define('param', param_name) # TODO: Params should have nodes too for Symbol decl?
        for stmt in func.body:
            resolve_stmt(stmt, func_scope)

def resolve_stmt(stmt, scope):
    if isinstance(stmt, VarDecl):
        resolve_expr(stmt.expr, scope)
        scope.define('var', stmt.name, stmt)
    elif isinstance(stmt, Return):
        resolve_expr(stmt.expr, scope)
    elif isinstance(stmt, ExprStmt):
        resolve_expr(stmt.expr, scope)
    else:
        raise Exception(f"Unknown stmt: {stmt}")

def resolve_expr(expr, scope):
    if isinstance(expr, Identifier):
        symbol = scope.resolve(expr.name)
        expr.resolved = symbol
    elif isinstance(expr, Number):
        pass
    elif isinstance(expr, String):
        pass
    elif isinstance(expr, BinaryOp):
        resolve_expr(expr.left, scope)
        resolve_expr(expr.right, scope)
    elif isinstance(expr, Call):
        symbol = scope.resolve(expr.func_name) # functions are in global scope
        expr.resolved = symbol
        for arg in expr.args:
            resolve_expr(arg, scope)

def build_exported_scope(module):
    scope = Scope()
    # For now only functions are imported
    for func in module.funcs:
        scope.define('func', func.name, func)
    return scope

### CODEGEN

def gen_c_header(module):
    header = ""
    header += f"#ifndef {module.name.upper()}_H\n#define {module.name.upper()}_H\n\n"
    for func in module.funcs:
        header += gen_function_header(func) + "\n"
    header += f"\n#endif\n"
    return header

def gen_c_source(module):
    source = ""
    source += f'#include "{module.name}.h"\n\n'
    for extern_block in module.extern_blocks:
        source += f'#include <{extern_block.header_name}>\n'
    source += "\n"
    for imp in module.imports:
        source += f'#include "{imp}.h"\n'
    source += "\n"
    for func in module.funcs:
        source += gen_function_body(func) + "\n\n"
    return source

def gen_function_header(func):
    params = ", ".join(f"{typ} {name}" for typ, name in func.params)
    return f"{func.return_type} {func.name}({params});"

def gen_function_body(func):
    header = gen_function_header(func).rstrip(";")
    body = "\n".join("    " + gen_stmt(stmt) for stmt in func.body)
    return f"{header}\n{{\n{body}\n}}"

def gen_stmt(stmt):
    if isinstance(stmt, Return):
        return f"return {gen_expr(stmt.expr)};"

    if isinstance(stmt, VarDecl):
        return f"{stmt.var_type} {stmt.name} = {gen_expr(stmt.expr)};"

    if isinstance(stmt, ExprStmt):
        return f"{gen_expr(stmt.expr)};"

    raise NotImplementedError("Unknown stmt")

def gen_expr(expr):
    if isinstance(expr, BinaryOp):
        return f"{gen_expr(expr.left)} {expr.op} {gen_expr(expr.right)}"

    if isinstance(expr, Call):
        args = ", ".join(gen_expr(arg) for arg in expr.args)
        return f"{expr.func_name}({args})"

    if isinstance(expr, Identifier):
        return expr.name

    if isinstance(expr, Number):
        return str(expr.value)

    if isinstance(expr, String):
        return f'"{expr.value}"'

    if isinstance(expr, str) and expr.startswith('"'):
        return expr

    raise NotImplementedError(f"Unknown expr: {expr}")

if __name__ == "__main__":
    main()