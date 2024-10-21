from pyformlang.cfg import CFG, Production, Epsilon


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cnf = cfg.to_normal_form()
    cnf_productions = cnf.productions

    cfg_nullable_symbols = cfg.get_nullable_symbols()

    wcnf_productions = set()

    for production in cnf_productions:
        wcnf_productions.add(production)

    for sym in cfg_nullable_symbols:
        wcnf_productions.add(Production(sym, [Epsilon()], filtering=False))

    return CFG(start_symbol=cfg.start_symbol, productions=wcnf_productions)
