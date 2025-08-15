import os
import shutil
import subprocess

def main():
    build("example", "example")
    # transpile_module("example/main.s2", "out")


def build(dir, prog_name):
    # Parse modules
    modules = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            modules.append(transpile_module(filepath))

    print("Modules parsed: " + ", ".join(module.name for module in modules))
    print_ast(modules)

    # Codegen modules into C files
    # out_path = os.path.join(dir, "c_out")
    # shutil.rmtree(out_path, ignore_errors=True)
    # os.makedirs(out_path)
    
    # for module in modules:
    #     gen_module(module, out_path)

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

def transpile_module(filepath):
    with open(filepath) as f:
        src = f.read()
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    module = parser.parse_module()
    return module

def gen_module(module, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{module.name}.h"), "w") as h:
        h.write(f"#ifndef {module.name.upper()}_H\n#define {module.name.upper()}_H\n\n")
        for func in module.funcs:
            h.write(gen_function_header(func) + "\n")
        h.write(f"\n#endif\n")
    with open(os.path.join(out_dir, f"{module.name}.c"), "w") as c:
        c.write(f'#include "{module.name}.h"\n\n')
        for inc in module.includes:
            c.write(f'#include {inc}\n')
        c.write("\n")
        for imp in module.imports:
            c.write(f'#include "{imp}.h"\n')
        c.write("\n")
        for func in module.funcs:
            c.write(gen_function_body(func) + "\n\n")

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
        if ident in {'int', 'return', 'module', 'import'}:
            return Token(ident.upper(), ident)
        return Token('IDENT', ident)

    def match_number(self, first):
        num = first
        while self.peek().isdigit():
            num += self.advance()
        return Token('NUMBER', num)

    def read_string(self):
        result = '"'
        while True:
            ch = self.advance()
            result += ch
            if ch == '"':
                break
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
        
        mapping = {
            '(': 'LPAREN', ')': 'RPAREN',
            '{': 'LBRACE', '}': 'RBRACE',
            ',': 'COMMA', ';': 'SEMICOLON',
            '+': 'PLUS', '=': 'EQUAL',
            '<': 'LT', '>': 'GT',
            '.': 'PERIOD', '#': 'HASH',
        }

        if ch in mapping:
            return Token(mapping[ch], ch)
        
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
    def __init__(self, name):
        self.name = name

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
    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args
    
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

class Module(Node):
    def __init__(self, name, imports, includes, funcs):
        self.name = name
        self.imports = imports
        self.includes = includes
        self.funcs = funcs

    def __repr__(self):
        return f"Module({self.name}, {self.imports}, {self.includes}, {self.funcs})"

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

    def expect(self, kind):
        token = self.advance()
        if token.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {token.kind}");
        return token

    def parse_module(self):
        self.expect('MODULE')
        name = self.expect('IDENT').value
        self.expect('SEMICOLON')

        imports = []
        includes = []
        funcs = []

        while self.peek().kind != 'EOF':
            if self.peek().kind == 'IMPORT':
                self.advance()
                imports.append(self.expect('IDENT').value)
                self.expect('SEMICOLON')
            elif self.peek().kind == 'HASH':
                self.advance()
                if self.expect('IDENT').value != 'include':
                    raise SyntaxError("Expected include")
                header = self.expect('LT').value + self.expect('IDENT').value
                while self.peek().kind != 'GT':
                    header += self.advance().value
                header += self.expect('GT').value
                includes.append(header)
            else:
                funcs.append(self.parse_function())

        return Module(name, imports, includes, funcs)

    def parse_function(self):
        ret_type = self.expect('INT').value
        name = self.expect('IDENT').value
        self.expect('LPAREN')

        params = []
        while self.peek().kind != 'RPAREN':
            type_tok = self.expect('INT')
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
            elif tok.kind == 'INT':
                stmts.append(self.parse_var_decl())
            elif tok.kind == 'IDENT':
                stmts.append(self.parse_call_stmt())
        return stmts

    def parse_return(self):
        self.expect('RETURN')
        expr = self.parse_expr()
        self.expect('SEMICOLON')
        return Return(expr)

    def parse_var_decl(self):
        var_type = self.expect('INT').value
        name = self.expect('IDENT').value
        self.expect('EQUAL')
        expr = self.parse_expr()
        self.expect('SEMICOLON')
        return VarDecl(var_type, name, expr)

    def parse_call_stmt(self):
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
        for inc in node.includes:
            print(f"{pad}  Include({inc})")
        for func in node.funcs:
            print_ast(func, indent + 1)
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
    elif isinstance(node, BinaryOp):
        print(f"{pad}BinaryOp {node.op}")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, Identifier):
        print(f"{pad}Identifer {node.name}")
    elif isinstance(node, Number):
        print(f"{pad}Number {node.value}")
    elif isinstance(node, String):
        print(f"{pad}String {node.value}")
    else:
        print(f"{pad}Unknown {node}")


### CODEGEN

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
        return expr.value # already quoted

    if isinstance(expr, str) and expr.startswith('"'):
        return expr

    raise NotImplementedError(f"Unknown expr: {expr}")

if __name__ == "__main__":
    main()