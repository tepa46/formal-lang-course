from antlr4 import (
    InputStream,
    CommonTokenStream,
    ParserRuleContext,
)

from project.qgl.QGL_grammarLexer import QGL_grammarLexer
from project.qgl.QGL_grammarParser import QGL_grammarParser


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    input_stream = InputStream(program)
    lexer = QGL_grammarLexer(input_stream)
    stream = CommonTokenStream(lexer)

    parser = QGL_grammarParser(stream)
    parser.removeParseListeners()

    tree = parser.prog()

    if parser.getNumberOfSyntaxErrors() != 0:
        return None, False

    return tree, True


def nodes_count(tree: ParserRuleContext) -> int:
    result = 1

    if tree.getChildCount() == 0:
        return result

    for child_tree in tree.getChildren():
        result += nodes_count(child_tree)

    return result


def tree_to_program(tree: ParserRuleContext) -> str:
    if tree.getChildCount() == 0:
        new_text = tree.getText()
        return f"{new_text} " if new_text != "\n" else new_text

    result = ""

    for child_tree in tree.getChildren():
        result += tree_to_program(child_tree)

    return result
