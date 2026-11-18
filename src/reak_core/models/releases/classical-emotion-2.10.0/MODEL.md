# Fiche modèle — classical-emotion 2.10.0

Paquet publié. L'alias de production reste classical-emotion 2.6.0.
Chargement : `SessionNeutralModel.from_release_dir`.

## Données

Jeu 2.9.0 plus HERMES_FS002_S02–S07 (chaque enregistrement = une unité LOSO).
Exclus : P004, P007, P008, P009, P013, P018. Plancher mean RMS 6 sur SADNESS et FEAR.
CONTEMPT : moitié LEFT + moitié RIGHT. Coupe-bande 60 Hz (Q = 30) puis 20–110 Hz.

## Limites

LOSO imbriqué 0,476 [0,409, 0,545]. Sur Malefycia, NEUTRAL hors ligne ≈ 0,301 contre 0,364 pour 2.6.0.
Pas de lissage a posteriori dans ce paquet. FS002 est en bootstrap d'instruction tant qu'une revue visage n'existe pas.
