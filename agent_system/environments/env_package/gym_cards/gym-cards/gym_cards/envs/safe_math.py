"""Safe evaluation for the small arithmetic expressions used by Gym Cards."""

from __future__ import annotations

import ast
import math
from fractions import Fraction


class UnsafeExpression(ValueError):
    """Raised when an expression contains unsupported syntax or values."""


def evaluate_arithmetic(expression: str) -> Fraction:
    """Evaluate ``+``, ``-``, ``*``, ``/`` and parentheses without ``eval``.

    The environment receives expressions assembled from model actions.  Parsing
    an allow-listed AST keeps that input data-only and prevents calls,
    attributes, comprehensions, or other Python syntax from executing.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise UnsafeExpression("invalid arithmetic expression") from exc

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(float(node.value)):
                raise UnsafeExpression("non-finite number")
            value = Fraction(node.value)
            if abs(value) > 10**9:
                raise UnsafeExpression("number is too large")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                if right == 0:
                    raise UnsafeExpression("division by zero")
                value = left / right
            if abs(value) > 10**9:
                raise UnsafeExpression("result is too large")
            return value
        raise UnsafeExpression("unsupported arithmetic syntax")

    return visit(tree)
