OPENING_BOOK = {
    # ———————————————————————————
    # Position initiale (Blancs au trait)
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': [
        'e4',   # Italienne, Espagnole, Sicilienne, Française, Scandinave, Alekhine, Pirc, Gambit Roi
        'd4',   # Gambit Dame, Indienne Roi, Grünfeld
        'c4',   # Anglaise
        'Nf3',  # Réti
        'f4',   # Gambit Roi
    ],

    # ———————————————————————————
    # Noirs contre 1.e4
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1': [
        'e5',   # Italienne, Espagnole
        'c5',   # Sicilienne
        'e6',   # Française
        'd5',   # Scandinave
        'Nf6',  # Alekhine
        'd6',   # Pirc
    ],

    # Positions après 1.e4 e5 2.Nf3
    'rnbqkbnr/pppppppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2': [
        'Bc4',  # Italienne
        'Bb5',  # Espagnole
    ],

    # Positions après 1.e4 c5 2.Nf3
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2': [
        'd4',   # Sicilienne ouverte
    ],

    # Positions après 1.e4 e6 2.d4 d5 3.e5
    'rnbqkbnr/ppp1pppp/8/3pP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3': [
        'Nc3',  # Française
    ],

    # Positions après 1.e4 d5 2.exd5 Qxd5 3.Nc3
    'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 3': [
        'Qa5',  # Scandinave
    ],

    # Positions après 1.e4 Nf6 2.e5 Nd5 3.d4
    'rnbqkb1r/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 1 3': [
        'd6',  # Alekhine
    ],

    # Positions après 1.e4 d6 2.d4 Nf6 3.Nc3
    'rnbqkb1r/pppppppp/3p4/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3': [
        'g6',  # Pirc
    ],

    # ———————————————————————————
    # Noirs contre 1.d4
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1': [
        'd5',   # Gambit Dame, Indienne du Roi, Grünfeld
        'Nf6',  # Indienne Roi
        'e6',   # Indienne de Dame
        'g6',   # Indienne Roi
        'c5',   # Benoni
    ],

    # Positions après 1.d4 d5 2.c4
    'rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2': [
        # Gambit Dame
        'cxd5', # Accepter le gambit
    ],

    # Positions après 1.d4 Nf6 2.c4 g6 3.Nc3 d5
    'rnbqkb1r/ppp1pppp/6p1/3p4/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 1 3': [
        # Grünfeld
        'cxd5',
    ],

    # ———————————————————————————
    # Noirs contre 1.c4 (Anglaise)
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1': [
        'e5',   # Symétrique
        'c5',   # Contre-Anglaise
        'Nf6',  # Flexible
        'e6',   # Fermé
        'g6',   # Moderne / Fianchetto
        'd5',   # Centre direct
    ],

    # ———————————————————————————
    # Noirs contre 1.Nf3 (Réti)
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1': [
        'd5',   # Centre
        'Nf6',  # Flexible
        'g6',   # Moderne / Fianchetto
        'e6',   # Fermé
        'c5',   # Contre-coup de flanc
    ],

    # ———————————————————————————
    # Positions après 1.e4 e5 2.f4
    # ———————————————————————————
    'rnbqkbnr/pppppppp/8/4p3/5P2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2': [
        'exf4', # Accepter le gambit
        'd5',   # Refuser et contrer au centre
        'Nc6',  # Développement
    ],
}

