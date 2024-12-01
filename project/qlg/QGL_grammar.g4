grammar QGL_grammar;

prog
    : EOL* (stmt EOL*)* EOF
    ;

stmt
    : bind
    | add
    | remove
    | declare
    ;

declare
    : LET VAR IS GRAPH
    ;

bind
    : LET VAR EQUALS expr
    ;

remove
    : REMOVE (VERTEX | EDGE | VERTICES) expr FROM VAR
    ;

add
    : ADD (VERTEX | EDGE) expr TO VAR
    ;

expr
    : NUM
    | CHAR
    | VAR
    | edge_expr
    | set_expr
    | regexp
    | select
    ;

set_expr
    : LBRACKET expr (COMMA expr)* RBRACKET
    ;

edge_expr
    : LPAREN expr COMMA expr COMMA expr RPAREN
    ;

regexp
    : CHAR
    | VAR
    | LPAREN regexp RPAREN
    | regexp VBAR regexp
    | regexp CARET range
    | regexp DOT regexp
    | regexp AMPER regexp
    ;

range
    : LBRACKET NUM DDOT NUM? RBRACKET
    ;

select
    : v_filter? v_filter? RETURN VAR (COMMA VAR)? WHERE VAR REACHABLE FROM VAR IN VAR BY expr
    ;

v_filter
    : FOR VAR IN expr
    ;

LET
    : 'let'
    ;

IS
    : 'is'
    ;

FROM
    : 'from'
    ;

TO
    : 'to'
    ;

IN
    : 'in'
    ;

BY
    : 'by'
    ;

RETURN
    : 'return'
    ;

FOR
    : 'for'
    ;

WHERE
    : 'where'
    ;

REACHABLE
    : 'reachable'
    ;

GRAPH
    : 'graph'
    ;

VERTEX
    : 'vertex'
    ;

VERTICES
    : 'vertices'
    ;

EDGE
    : 'edge'
    ;

ADD
    : 'add'
    ;

REMOVE
    : 'remove'
    ;

EQUALS
    : '='
    ;

LPAREN
    : '('
    ;

RPAREN
    : ')'
    ;

LBRACKET
    : '['
    ;

RBRACKET
    : ']'
    ;

COMMA
    : ','
    ;

AMPER
    : '&'
    ;

VBAR
    : '|'
    ;

DOT
    : '.'
    ;

DDOT
    : '..'
    ;

CARET
    : '^'
    ;

VAR
    : [a-z][a-z0-9]*
    ;

NUM
    : '0' | [1-9][0-9]*
    ;

CHAR
    : '"'[a-z]'"'
    ;

EOL
    : '\n'
    ;

WS
    : [ \t\r] -> skip
    ;
