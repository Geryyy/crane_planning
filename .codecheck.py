"""Prove that only comments/docstrings changed: compare AST (docstrings stripped)."""
import ast, subprocess, sys, io

def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
               and isinstance(b[0].value.value, str):
                if len(b) > 1:
                    node.body = b[1:]
                else:
                    node.body = [ast.Pass()]
    return tree

def norm(src):
    return ast.dump(strip_docstrings(ast.parse(src)), annotate_fields=True)

def old(path, ref):
    return subprocess.run(['git', 'show', f'{ref}:{path}'], capture_output=True, text=True,
                          cwd='/workspaces/ros2_baustelle_ws/src/concrete_block_stack/crane_planning').stdout

ref = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'
files = subprocess.run(['git', 'diff', '--name-only', ref], capture_output=True, text=True,
                       cwd='/workspaces/ros2_baustelle_ws/src/concrete_block_stack/crane_planning').stdout.split()
bad = 0
for f in files:
    if f.endswith('.py'):
        try:
            a, b = norm(old(f, ref)), norm(open(f).read())
        except SyntaxError as e:
            print('SYNTAX ERROR', f, e); bad += 1; continue
        if a != b:
            print('CODE CHANGED:', f); bad += 1
        else:
            print('ok', f)
    elif f.endswith(('.yaml', '.yml')):
        import yaml
        if yaml.safe_load(old(f, ref)) != yaml.safe_load(open(f)):
            print('YAML DATA CHANGED:', f); bad += 1
        else:
            print('ok', f)
    else:
        print('?? (manual review)', f)
print('FAIL' if bad else 'ALL CLEAN')
sys.exit(1 if bad else 0)
