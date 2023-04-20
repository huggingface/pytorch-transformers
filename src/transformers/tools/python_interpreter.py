import ast
from typing import Any, Callable, Dict


class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


def evaluate(code: str, tools: Dict[str, Callable]):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    expression = ast.parse(code)
    state = {}
    result = None
    for node in expression.body:
        result = evaluate_ast(node, state, tools)

    if result is None:
        # Try to find a result in the state
        if "result" in state:
            return state["result"]
        for key in state:
            if "result" in key:
                return state[key]

        return state
    else:
        return result


def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]):
    """
    Evaluate an absract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abastract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    if isinstance(expression, ast.Assign):
        # Assignement -> we evaluate the assignement which should update the state
        evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        evaluate_if(expression, state, tools)
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return state[expression.id]
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, state, tools)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpretorError(f"{expression.__class__.__name__} is not supported.")


def evaluate_assign(assign, state, tools):
    var_names = assign.targets
    result = evaluate_ast(assign.value, state, tools)

    if len(var_names) == 1:
        state[var_names[0].id] = result
    else:
        if len(result) != len(var_names):
            raise InterpretorError(f"Expected {len(var_names)} values but got {len(result)}.")
        for var_name, r in zip(var_names, result):
            state[var_name.id] = r


def evaluate_call(call, state, tools):
    func_name = call.func.id
    if func_name not in tools:
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func.id})."
        )

    func = tools[func_name]
    # Todo deal with args
    args = {evaluate_ast(arg, state, tools) for arg in call.args}
    kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}
    return func(*args, **kwargs)


def evaluate_subscript(subscript, state, tools):
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)
    # TODO: add some logic here to fix typos of the LLM.
    return value[index]


def evaluate_condition(condition, state, tools):
    if len(condition.ops) > 1:
        raise InterpretorError("Cannot evaluate conditions with multiple operators")

    left = evaluate_ast(condition.left, state, tools)
    comparator = condition.ops[0]
    right = evaluate_ast(condition.comparators[0], state, tools)

    if isinstance(comparator, ast.Eq):
        return left == right
    elif isinstance(comparator, ast.NotEq):
        return left != right
    elif isinstance(comparator, ast.Lt):
        return left < right
    elif isinstance(comparator, ast.LtE):
        return left <= right
    elif isinstance(comparator, ast.Gt):
        return left > right
    elif isinstance(comparator, ast.GtE):
        return left >= right
    elif isinstance(comparator, ast.Is):
        return left is right
    elif isinstance(comparator, ast.IsNot):
        return left is not right
    elif isinstance(comparator, ast.In):
        return left in right
    elif isinstance(comparator, ast.NotIn):
        return left not in right
    else:
        raise InterpretorError(f"Operator not supported: {comparator}")


def evaluate_if(if_statement, state, tools):
    if evaluate_condition(if_statement.test, state, tools):
        for line in if_statement.body:
            evaluate_ast(line, state, tools)
    else:
        for line in if_statement.orelse:
            evaluate_ast(line, state, tools)
