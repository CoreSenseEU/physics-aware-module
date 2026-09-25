import ast

import SRConfig as SRConfig

# Supported binary operators
ALL_OPERATORS = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**'
}

def collect_used_operators_and_functions(node, used_ops, used_funcs):
    """Recursively collect all used operators and functions."""
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in ALL_OPERATORS:
            used_ops.add(ALL_OPERATORS[op_type])
        collect_used_operators_and_functions(node.left, used_ops, used_funcs)
        collect_used_operators_and_functions(node.right, used_ops, used_funcs)

    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            used_ops.add('U')
        collect_used_operators_and_functions(node.operand, used_ops, used_funcs)

    elif isinstance(node, ast.Call):
        func_name = get_func_name(node.func)
        used_funcs.add(func_name)
        for arg in node.args:
            collect_used_operators_and_functions(arg, used_ops, used_funcs)

    elif isinstance(node, (ast.Constant, ast.Name)):
        return

def get_func_name(func_node):
    """Return simplified function name, like 'sin' from 'np.sin'."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    elif isinstance(func_node, ast.Attribute):
        return func_node.attr
    raise ValueError("Unsupported function type")

def is_constant(node):
    try:
        const_number(node)
        return True
    except TypeError:
        return False

def _is_add(node):
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)

def const_number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = const_number(node.operand)
        return -val if isinstance(node.op, ast.USub) else val

    raise TypeError(f"Not a numeric constant: {type(node).__name__}")

def _term_var_signature(n):
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        return _term_var_signature(n.operand)

    if isinstance(n, ast.Name) and n.id.startswith("x"):
        return (n.id,)

    if is_constant(n):
        return tuple()

    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        ls = _term_var_signature(n.left)
        rs = _term_var_signature(n.right)
        if ls is None or rs is None:
            return None
        sig = tuple(sorted(ls + rs))
        return sig

    return None

def prune_small_constants(node, threshold=SRConfig.simplificationWeightThreshold, in_mult=False):
    """
    - If abs(const) < threshold:
        - inside multiplication (in_mult=True): erase whole multiplication subtree (return None)
        - otherwise: drop the constant term (return None)
    """
    if node is None:
        return None

    try:
        v = const_number(node)
        if abs(v) < threshold:
            return None
        return node
    except TypeError:
        pass

    if isinstance(node, ast.Name):
        return node

    if isinstance(node, ast.Constant):
        return node

    if isinstance(node, ast.UnaryOp):
        op = prune_small_constants(node.operand, threshold, in_mult=in_mult)
        if op is None:
            return None
        return ast.UnaryOp(op=node.op, operand=op)

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):
            l = prune_small_constants(node.left, threshold, in_mult=True)
            if l is None:
                return None
            r = prune_small_constants(node.right, threshold, in_mult=True)
            if r is None:
                return None
            return ast.BinOp(left=l, op=node.op, right=r)

        if isinstance(node.op, ast.Add):
            l = prune_small_constants(node.left, threshold, in_mult=False)
            r = prune_small_constants(node.right, threshold, in_mult=False)
            if l is None and r is None:
                return None
            if l is None:
                return r
            if r is None:
                return l
            return ast.BinOp(left=l, op=node.op, right=r)

        if isinstance(node.op, ast.Sub):
            l = prune_small_constants(node.left, threshold, in_mult=False)
            r = prune_small_constants(node.right, threshold, in_mult=False)
            if l is None and r is None:
                return None
            if r is None:
                return l
            if l is None:
                return ast.UnaryOp(op=ast.USub(), operand=r)
            return ast.BinOp(left=l, op=node.op, right=r)

        l = prune_small_constants(node.left, threshold, in_mult=False)
        r = prune_small_constants(node.right, threshold, in_mult=False)
        if l is None or r is None:
            return None
        return ast.BinOp(left=l, op=node.op, right=r)

    if isinstance(node, ast.Call):
        new_args = []
        for a in node.args:
            ap = prune_small_constants(a, threshold, in_mult=False)
            if ap is None:
                return None
            new_args.append(ap)
        return ast.Call(func=node.func, args=new_args, keywords=node.keywords)

    return node

def to_prefix(node, allowed_ops, allowed_funcs):
    def _collect_addsub_terms(n):
        terms = []

        def rec(x, sign=1):
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
                rec(x.left, sign)
                rec(x.right, sign)
            elif isinstance(x, ast.BinOp) and isinstance(x.op, ast.Sub):
                rec(x.left, sign)
                rec(x.right, -sign)
            else:
                terms.append((sign, x))

        rec(n, 1)
        return terms

    def _negate_term_node(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            if is_constant(n.left):
                new_left = ast.UnaryOp(op=ast.USub(), operand=n.left)
                return ast.BinOp(left=new_left, op=ast.Mult(), right=n.right)
            if is_constant(n.right):
                new_right = ast.UnaryOp(op=ast.USub(), operand=n.right)
                return ast.BinOp(left=n.left, op=ast.Mult(), right=new_right)
        return ast.UnaryOp(op=ast.USub(), operand=n)

    def _term_var_signature(n):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return _term_var_signature(n.operand)

        if isinstance(n, ast.Name) and n.id.startswith("x"):
            return (n.id,)

        if is_constant(n):
            return tuple()

        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            ls = _term_var_signature(n.left)
            rs = _term_var_signature(n.right)
            if ls is None or rs is None:
                return None
            return tuple(sorted(ls + rs))

        return None

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        op_symbol = ALL_OPERATORS.get(op_type)

        if op_symbol in ('+', '-'):
            terms = _collect_addsub_terms(node)

            const_value = 0.0
            saw_const = False
            signed_nonconst = []

            for sign, t in terms:
                if is_constant(t):
                    saw_const = True
                    const_value += sign * const_number(t)
                else:
                    signed_nonconst.append((sign, t))

            items = []
            for sign, t in signed_nonconst:
                sig = _term_var_signature(t)
                t2 = _negate_term_node(t) if sign == -1 else t
                pref = to_prefix(t2, allowed_ops, allowed_funcs)

                if sig is None:
                    sig = ("~", str(pref))

                items.append((sig, sign, str(pref)))

            items.sort(key=lambda x: (x[0], x[1] == -1, x[2]))
            nonconst_prefixes = [p for _, _, p in items]

            if not nonconst_prefixes:
                return str(const_value)

            if saw_const and const_value != 0.0:
                const_node = ast.Constant(value=const_value)
                const_pref = to_prefix(const_node, allowed_ops, allowed_funcs)
                acc = f"B {nonconst_prefixes[0]} {const_pref}"
                start = 1
            else:
                acc = nonconst_prefixes[0]
                start = 1

            for p in nonconst_prefixes[start:]:
                acc = f"+ {acc} {p}"
            return acc

        left_node = node.left
        right_node = node.right
        left = to_prefix(left_node, allowed_ops, allowed_funcs)
        right = to_prefix(right_node, allowed_ops, allowed_funcs)

        if op_symbol == '*':
            if is_constant(left_node) or is_constant(right_node):
                return f"* {left} {right}"
            else:
                return f"m {left} {right}"

        return f"{op_symbol} {left} {right}"

    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            operand = to_prefix(node.operand, allowed_ops, allowed_funcs)
            return f"U {operand}"
        if isinstance(node.op, ast.UAdd):
            return to_prefix(node.operand, allowed_ops, allowed_funcs)

    elif isinstance(node, ast.Call):
        func_name = get_func_name(node.func)
        args = [to_prefix(arg, allowed_ops, allowed_funcs) for arg in node.args]
        return f"{func_name} {' '.join(args)}"

    elif isinstance(node, ast.Constant):
        return str(node.value)

    elif isinstance(node, ast.Name):
        return node.id

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def rewrite_using_available_operators(prefix_expr, available_operators):
    """
    Rewrites a subtree in prefix notation, replacing **2 or **3 with m (multiplication) if these operators are not in the available set.
    """
    tokens = prefix_expr.split()
    stack = []

    for token in reversed(tokens):
        if token == "**":
            base = stack.pop()
            exponent = stack.pop()
            if exponent == "2" and "**2" not in available_operators:
                stack.append(f"m {base} {base}")
            elif exponent == "3" and "**3" not in available_operators:
                if "**2" in available_operators:
                    stack.append(f"m ** {base} 2 {base}")
                else:
                    stack.append(f"m m {base} {base} {base}")
            else:
                stack.append(f"** {base} {exponent}")
        else:
            stack.append(token)

    return " ".join(reversed(stack))

def infix_to_prefix(expr, available_operators=None):
    tree = ast.parse(expr, mode='eval')
    pruned = prune_small_constants(tree.body, threshold=SRConfig.simplificationWeightThreshold)
    if pruned is None:
        return "0"
    used_ops = set()
    used_funcs = set()
    collect_used_operators_and_functions(pruned, used_ops, used_funcs)
    prefix_tree = to_prefix(pruned, used_ops, used_funcs)

    if available_operators is not None:
        prefix_tree = rewrite_using_available_operators(prefix_tree, available_operators)

    return prefix_tree

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.prefix = ""
        self.depth = 0

def build_tree(tokens, is_root=True):
    def wrap_with_mul_if_needed(child, parent_op, parent=None):
        if isinstance(child.value, str):
            if child.value in {'*', 'B', 'U'}:
                child.parent = parent
                return child
            if parent_op == '+' and child.value == '+':
                child.parent = parent
                return child
            if parent_op == 'B':
                child.parent = parent
                return child
        if isinstance(child.value, (int, float)):
            child.parent = parent
            return child
        mul_node = Node('*')
        mul_node.left = Node(1)
        mul_node.right = child
        mul_node.left.parent = mul_node
        mul_node.right.parent = mul_node
        mul_node.prefix = f"* 1 {child.prefix}"
        mul_node.parent = parent
        return mul_node

    if not tokens:
        return None

    token = tokens.pop(0)
    original_token = token

    if token == 'U':
        if not tokens:
            raise ValueError("Expected token after 'U'")
        node = Node('U')
        node.left = build_tree(tokens, is_root=False)
        node.left.parent = node
        node.prefix = f"{original_token} {node.left.prefix}"
        result = node

    elif token.replace('.', '', 1).isdigit():
        token = float(token) if '.' in token else int(token)
        node = Node(token)
        node.prefix = str(token)
        result = node

    else:
        node = Node(token)

        binary_ops = {'+', '-', '*', '/', '**', '^', 'B', 'm'}
        unary_ops = {'sin', 'cos', 'tanh', 'arctan', 'exp', 'log', 'abs', 'U'}

        if token in binary_ops:
            node.left = build_tree(tokens, is_root=False)
            node.right = build_tree(tokens, is_root=False)

            if token != '*':
                node.left = wrap_with_mul_if_needed(node.left, token, node)
                node.right = wrap_with_mul_if_needed(node.right, token, node)

            node.left.parent = node
            node.right.parent = node

            node.prefix = f"{original_token} {node.left.prefix} {node.right.prefix}"

        elif token in unary_ops:
            node.left = build_tree(tokens, is_root=False)
            node.left.parent = node
            if token != '*':
                node.left = wrap_with_mul_if_needed(node.left, token, node)
            node.prefix = f"{original_token} {node.left.prefix}"

        else:
            node.prefix = str(token)

        result = node

    if is_root and isinstance(result.value, str) and result.value not in  {'*', '+', 'B', 'U'}:
        result = wrap_with_mul_if_needed(result, result.value)
    elif is_root and result.value == 'B' and isinstance(result.left.value, str) and result.left.value not in {'*', '+'}:
        old_left = result.left
        new_left = Node('*')
        new_left.left = Node(1)
        new_left.right = old_left
        new_left.left.parent = new_left
        new_left.right.parent = new_left
        new_left.prefix = f"* 1 {old_left.prefix}"
        result.left = new_left
        new_left.parent = result
        result.prefix = f"B {new_left.prefix} {result.right.prefix}"

    if is_root:
        result.parent = None

    return result

def is_leaf(node):
    if isinstance(node.value, (int, float)):
        return True
    if isinstance(node.value, str) and node.value.startswith("x"):
        return True
    return False


def compute_depth(node):
    """Computes and sets the depth for each node recursively."""
    if node is None:
        return -1

    if is_leaf(node):
        node.depth = -1
        return -1

    left_depth = compute_depth(node.left)
    right_depth = compute_depth(node.right)
    if node.value in {'U'}:  # unary minus does not increase depth
        node.depth = max(left_depth, right_depth)
    elif node.value == '*':
        if node.parent is None or node.parent.value in {'*'}:  # it is not another operator
            node.depth = 1 + max(left_depth, right_depth)
        else:
            node.depth = max(left_depth, right_depth)
    elif node.value == '+':
        if (node.left and node.left.value == '+') or (node.right and node.right.value == '+'):  # ident with multiple inputs
            node.depth = max(left_depth, right_depth)
        else:
            node.depth = 1 + max(left_depth, right_depth)
    elif node.value == 'B':
        if node.parent is not None and node.parent.value in {'*'}:
            node.depth = 1 + max(left_depth, right_depth)
        else:
            node.depth = max(left_depth, right_depth)
    else:
        node.depth = 1 + max(left_depth, right_depth)
    return node.depth


def prefix_to_tree(expression):
    tokens = expression.split()
    tree = build_tree(tokens)
    compute_depth(tree)
    return tree


def count_operators(expression):
    """"
    Counts number of operators in the expression.
    """
    tokens = expression.split()
    count = 0
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]
        if token in {'-', '+'}:
            # when there are multiple '+' operators following, count it only once
            if i == 0 or tokens[i - 1] not in {'+', '-'}:
                count += 1
        elif token in {'B', 'U', '*'} or token.startswith('x') or token.replace('.', '', 1).isdigit():
            pass
        else:
            count += 1
        i += 1

    return count
